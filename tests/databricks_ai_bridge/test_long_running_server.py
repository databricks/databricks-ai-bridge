"""Tests for LongRunningAgentServer route registration, background handling, and SSE format."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from databricks_ai_bridge.long_running.server import (
    LongRunningAgentServer,
    _deferred_mark_failed,
    _sse_event,
)
from databricks_ai_bridge.long_running.settings import LongRunningSettings

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MODULE = "databricks_ai_bridge.long_running.server"


def _make_server(**kwargs):
    """Create a LongRunningAgentServer with DB disabled (no real Lakebase needed)."""
    with patch(f"{MODULE}.is_db_configured", return_value=False):
        return LongRunningAgentServer("ResponsesAgent", **kwargs)


def _mock_span():
    """Return a mock MLflow span with the attributes the server uses."""
    span = MagicMock()
    span.trace_id = "trace_abc123"
    span.set_inputs = MagicMock()
    span.set_outputs = MagicMock()
    span.set_attribute = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


def _mock_validator(server):
    """Patch the server's validator to pass through dicts unchanged."""
    server.validator = MagicMock()
    server.validator.validate_and_convert_result = MagicMock(side_effect=lambda x, **kw: x)


class TestSSEEvent:
    def test_dict_data(self):
        result = _sse_event("response.created", {"id": "resp_123", "status": "in_progress"})
        assert result.startswith("event: response.created\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["id"] == "resp_123"

    def test_string_data(self):
        result = _sse_event("error", "something went wrong")
        assert "event: error\n" in result
        assert "data: something went wrong\n\n" in result


class TestLongRunningSettings:
    def test_defaults(self):
        s = LongRunningSettings()
        assert s.task_timeout_seconds == 3600.0
        assert s.poll_interval_seconds == 1.0
        assert s.db_statement_timeout_ms == 5000
        assert s.cleanup_timeout_seconds == 7.0

    def test_validation_cleanup_must_exceed_db_timeout(self):
        with pytest.raises(ValueError, match="cleanup_timeout_seconds"):
            LongRunningSettings(db_statement_timeout_ms=5000, cleanup_timeout_seconds=4.0)

    def test_validation_positive(self):
        with pytest.raises(ValueError, match="task_timeout_seconds must be positive"):
            LongRunningSettings(task_timeout_seconds=-1)


class TestTransformStreamEvent:
    def test_default_is_noop(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer("ResponsesAgent")
        event = {"type": "response.output_item.done", "item": {"id": "fake_id"}}
        result = server.transform_stream_event(event, "resp_real")
        assert result is event

    def test_subclass_override(self):
        class CustomServer(LongRunningAgentServer):
            def transform_stream_event(self, event, response_id):
                if isinstance(event, dict):
                    return {
                        k: response_id if v == "FAKE" else v for k, v in event.items()
                    }
                return event

        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = CustomServer("ResponsesAgent")
        event = {"type": "response.created", "id": "FAKE"}
        result = server.transform_stream_event(event, "resp_real")
        assert result["id"] == "resp_real"
        assert result["type"] == "response.created"


class TestAgentTypeValidation:
    def test_rejects_non_responses_agent(self):
        with pytest.raises(ValueError, match="only supports 'ResponsesAgent'"):
            LongRunningAgentServer("ChatAgent")

    def test_accepts_responses_agent(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer("ResponsesAgent")
        assert server.agent_type == "ResponsesAgent"

    def test_default_agent_type(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer()
        assert server.agent_type == "ResponsesAgent"


class TestRouteRegistration:
    def test_routes_without_db(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer("ResponsesAgent")

        routes = [r.path for r in server.app.routes]
        # Parent routes should exist
        assert "/invocations" in routes
        # Retrieve endpoint should NOT be registered
        assert "/responses/{response_id}" not in routes

    def test_routes_with_db(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=True
        ):
            server = LongRunningAgentServer("ResponsesAgent")

        routes = [r.path for r in server.app.routes]
        assert "/responses/{response_id}" in routes


class TestDeferredMarkFailed:
    @pytest.mark.asyncio
    async def test_marks_response_failed(self):
        with patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            return_value=[(0, None, {"type": "response.created"})],
        ) as mock_get, patch(
            "databricks_ai_bridge.long_running.server.append_message",
            new_callable=AsyncMock,
        ) as mock_append, patch(
            "databricks_ai_bridge.long_running.server.update_response_status",
            new_callable=AsyncMock,
        ) as mock_update:
            await _deferred_mark_failed("resp_123", delay=0.01)

            mock_get.assert_awaited_once()
            mock_append.assert_awaited_once()
            args = mock_append.call_args
            assert args[0][0] == "resp_123"
            assert args[0][1] == 1  # next_seq after seq 0
            stream_event = args[1]["stream_event"]
            assert stream_event["type"] == "error"
            assert stream_event["error"]["code"] == "task_timeout"
            mock_update.assert_awaited_once_with("resp_123", "failed")

    @pytest.mark.asyncio
    async def test_handles_db_error_gracefully(self):
        with patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB down"),
        ):
            # Should not raise
            await _deferred_mark_failed("resp_123", delay=0.01)


class TestRetrieveRequest:
    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer("ResponsesAgent")

        with patch(
            "databricks_ai_bridge.long_running.server.get_response",
            new_callable=AsyncMock,
            return_value=None,
        ):
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await server._handle_retrieve_request("resp_missing", stream=False, starting_after=0)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_completed_returns_output(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer("ResponsesAgent")

        with patch(
            "databricks_ai_bridge.long_running.server.get_response",
            new_callable=AsyncMock,
            return_value=("resp_123", "completed", datetime.now(timezone.utc), "trace_abc"),
        ), patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            return_value=[
                (0, '{"text": "hi"}', {"type": "response.output_item.done", "item": {"text": "hi"}}),
            ],
        ):
            result = await server._handle_retrieve_request(
                "resp_123", stream=False, starting_after=0
            )
            assert result["id"] == "resp_123"
            assert result["status"] == "completed"
            assert result["output"] == [{"text": "hi"}]
            assert result["metadata"] == {"trace_id": "trace_abc"}

    @pytest.mark.asyncio
    async def test_stale_run_detection(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer(
                "ResponsesAgent", task_timeout_seconds=10.0
            )

        from datetime import timedelta

        old_time = datetime.now(timezone.utc) - timedelta(seconds=100)  # well past timeout
        with patch(
            "databricks_ai_bridge.long_running.server.get_response",
            new_callable=AsyncMock,
            return_value=("resp_stale", "in_progress", old_time, None),
        ), patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "databricks_ai_bridge.long_running.server.append_message",
            new_callable=AsyncMock,
        ) as mock_append, patch(
            "databricks_ai_bridge.long_running.server.update_response_status",
            new_callable=AsyncMock,
        ):
            result = await server._handle_retrieve_request(
                "resp_stale", stream=False, starting_after=0
            )
            assert result["status"] == "failed"
            mock_append.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_in_progress_returns_status(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer("ResponsesAgent")

        with patch(
            "databricks_ai_bridge.long_running.server.get_response",
            new_callable=AsyncMock,
            return_value=("resp_123", "in_progress", datetime.now(timezone.utc), None),
        ), patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await server._handle_retrieve_request(
                "resp_123", stream=False, starting_after=0
            )
            assert result == {"id": "resp_123", "status": "in_progress"}


class TestStreamRetrieve:
    @pytest.mark.asyncio
    async def test_completed_stream(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer(
                "ResponsesAgent", poll_interval_seconds=0.01
            )

        with patch(
            "databricks_ai_bridge.long_running.server.get_response",
            new_callable=AsyncMock,
            return_value=("resp_123", "completed", datetime.now(timezone.utc), None),
        ), patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            return_value=[
                (0, None, {"type": "response.created", "id": "resp_123"}),
                (1, '{"text": "hi"}', {"type": "response.output_item.done", "item": {"text": "hi"}}),
            ],
        ):
            events = []
            async for chunk in server._stream_retrieve("resp_123", starting_after=0):
                events.append(chunk)

            # Should have 2 SSE events + [DONE]
            assert len(events) == 3
            assert "response.created" in events[0]
            assert "response.output_item.done" in events[1]
            assert events[2] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_failed_stream_stops(self):
        with patch(
            "databricks_ai_bridge.long_running.server.is_db_configured", return_value=False
        ):
            server = LongRunningAgentServer(
                "ResponsesAgent", poll_interval_seconds=0.01
            )

        with patch(
            "databricks_ai_bridge.long_running.server.get_response",
            new_callable=AsyncMock,
            return_value=("resp_123", "failed", datetime.now(timezone.utc), None),
        ), patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            return_value=[
                (0, None, {"type": "error", "error": {"message": "boom"}}),
            ],
        ):
            events = []
            async for chunk in server._stream_retrieve("resp_123", starting_after=0):
                events.append(chunk)

            assert len(events) == 1
            assert "error" in events[0]


# ---------------------------------------------------------------------------
# P0: Background execution loops
# ---------------------------------------------------------------------------


class TestDoBackgroundStream:
    @pytest.mark.asyncio
    async def test_persists_events_and_completes(self):
        server = _make_server()
        _mock_validator(server)
        span = _mock_span()

        async def fake_stream(request_data):
            yield {"type": "response.created", "response": {"id": "resp_1"}}
            yield {"type": "response.output_text.delta", "delta": "hello"}
            yield {"type": "response.output_item.done", "item": {"text": "hello"}}

        with patch(f"{MODULE}.get_stream_function", return_value=fake_stream), \
             patch(f"{MODULE}.mlflow") as mock_mlflow, \
             patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append, \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update, \
             patch(f"{MODULE}.ResponsesAgent") as mock_ra:
            mock_mlflow.start_span.return_value = span
            mock_ra.responses_agent_output_reducer.return_value = {"output": []}

            state = {"seq": 0}
            await server._do_background_stream("resp_1", {"input": "hi"}, False, state)

            assert mock_append.await_count == 3
            # Verify sequence numbers 0, 1, 2
            seqs = [call.args[1] for call in mock_append.call_args_list]
            assert seqs == [0, 1, 2]
            # Verify state tracks final seq
            assert state["seq"] == 3
            mock_update.assert_awaited_once_with("resp_1", "completed")

    @pytest.mark.asyncio
    async def test_calls_transform_stream_event(self):
        transform_calls = []

        class TrackingServer(LongRunningAgentServer):
            def transform_stream_event(self, event, response_id):
                transform_calls.append((event, response_id))
                return {**event, "transformed": True}

        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = TrackingServer("ResponsesAgent")
        _mock_validator(server)
        span = _mock_span()

        async def fake_stream(request_data):
            yield {"type": "response.created"}
            yield {"type": "response.output_text.delta", "delta": "x"}

        with patch(f"{MODULE}.get_stream_function", return_value=fake_stream), \
             patch(f"{MODULE}.mlflow") as mock_mlflow, \
             patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append, \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock), \
             patch(f"{MODULE}.ResponsesAgent") as mock_ra:
            mock_mlflow.start_span.return_value = span
            mock_ra.responses_agent_output_reducer.return_value = {"output": []}

            state = {"seq": 0}
            await server._do_background_stream("resp_t", {"input": "hi"}, False, state)

            assert len(transform_calls) == 2
            # Each call gets the correct response_id
            assert all(rid == "resp_t" for _, rid in transform_calls)
            # The transformed event is what gets persisted
            for call in mock_append.call_args_list:
                evt = call.kwargs.get("stream_event") or call.args[3] if len(call.args) > 3 else call.kwargs.get("stream_event")
                assert evt.get("transformed") is True

    @pytest.mark.asyncio
    async def test_no_stream_fn_marks_failed(self):
        server = _make_server()

        with patch(f"{MODULE}.get_stream_function", return_value=None), \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update:
            state = {"seq": 0}
            with pytest.raises(RuntimeError, match="No stream function registered"):
                await server._do_background_stream("resp_x", {}, False, state)
            mock_update.assert_awaited_once_with("resp_x", "failed")

    @pytest.mark.asyncio
    async def test_persists_trace_id_when_requested(self):
        server = _make_server()
        _mock_validator(server)
        span = _mock_span()

        async def fake_stream(request_data):
            yield {"type": "response.output_item.done", "item": {"text": "hi"}}

        with patch(f"{MODULE}.get_stream_function", return_value=fake_stream), \
             patch(f"{MODULE}.mlflow") as mock_mlflow, \
             patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append, \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock), \
             patch(f"{MODULE}.ResponsesAgent") as mock_ra:
            mock_mlflow.start_span.return_value = span
            mock_ra.responses_agent_output_reducer.return_value = {"output": []}

            state = {"seq": 0}
            await server._do_background_stream("resp_tr", {"input": "hi"}, True, state)

            # Last append_message should contain trace_id
            last_call = mock_append.call_args_list[-1]
            trace_evt = last_call.kwargs.get("stream_event")
            assert trace_evt == {"trace_id": "trace_abc123"}


class TestDoBackgroundInvoke:
    @pytest.mark.asyncio
    async def test_persists_output_items_and_completes(self):
        server = _make_server()
        _mock_validator(server)
        span = _mock_span()

        async def fake_invoke(request_data):
            return {
                "output": [
                    {"type": "message", "content": "hello"},
                    {"type": "message", "content": "world"},
                ]
            }

        with patch(f"{MODULE}.get_invoke_function", return_value=fake_invoke), \
             patch(f"{MODULE}.mlflow") as mock_mlflow, \
             patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append, \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update, \
             patch(f"{MODULE}.update_response_trace_id", new_callable=AsyncMock):
            mock_mlflow.start_span.return_value = span

            state = {"seq": 0}
            await server._do_background_invoke("resp_inv", {"input": "hi"}, False, state)

            assert mock_append.await_count == 2
            # Verify sequence numbers
            seqs = [call.args[1] for call in mock_append.call_args_list]
            assert seqs == [0, 1]
            # Verify each item is wrapped as response.output_item.done
            for call in mock_append.call_args_list:
                evt = call.kwargs["stream_event"]
                assert evt["type"] == "response.output_item.done"
                assert "item" in evt
            assert state["seq"] == 2
            mock_update.assert_awaited_once_with("resp_inv", "completed")

    @pytest.mark.asyncio
    async def test_trace_id_persisted_when_requested(self):
        server = _make_server()
        _mock_validator(server)
        span = _mock_span()

        async def fake_invoke(request_data):
            return {"output": [{"type": "message", "content": "done"}]}

        with patch(f"{MODULE}.get_invoke_function", return_value=fake_invoke), \
             patch(f"{MODULE}.mlflow") as mock_mlflow, \
             patch(f"{MODULE}.append_message", new_callable=AsyncMock), \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock), \
             patch(f"{MODULE}.update_response_trace_id", new_callable=AsyncMock) as mock_trace:
            mock_mlflow.start_span.return_value = span

            state = {"seq": 0}
            await server._do_background_invoke("resp_inv", {"input": "hi"}, True, state)

            mock_trace.assert_awaited_once_with("resp_inv", "trace_abc123")

    @pytest.mark.asyncio
    async def test_no_invoke_fn_marks_failed(self):
        server = _make_server()

        with patch(f"{MODULE}.get_invoke_function", return_value=None), \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update:
            state = {"seq": 0}
            with pytest.raises(RuntimeError, match="No invoke function registered"):
                await server._do_background_invoke("resp_x", {}, False, state)
            mock_update.assert_awaited_once_with("resp_x", "failed")

    @pytest.mark.asyncio
    async def test_sync_invoke_fn_supported(self):
        server = _make_server()
        _mock_validator(server)
        span = _mock_span()

        def sync_invoke(request_data):
            return {"output": [{"type": "message", "content": "sync"}]}

        with patch(f"{MODULE}.get_invoke_function", return_value=sync_invoke), \
             patch(f"{MODULE}.mlflow") as mock_mlflow, \
             patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append, \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update, \
             patch(f"{MODULE}.update_response_trace_id", new_callable=AsyncMock):
            mock_mlflow.start_span.return_value = span

            state = {"seq": 0}
            await server._do_background_invoke("resp_sync", {"input": "hi"}, False, state)

            assert mock_append.await_count == 1
            mock_update.assert_awaited_once_with("resp_sync", "completed")


# ---------------------------------------------------------------------------
# P1: _task_scope error handling
# ---------------------------------------------------------------------------


class TestTaskScope:
    @pytest.mark.asyncio
    async def test_timeout_schedules_deferred_mark_failed(self):
        server = _make_server(task_timeout_seconds=0.01, cleanup_timeout_seconds=6.0)

        with patch(f"{MODULE}._deferred_mark_failed", new_callable=AsyncMock) as mock_deferred, \
             patch(f"{MODULE}.asyncio.create_task") as mock_create_task:
            state = {"seq": 0}
            async with server._task_scope("resp_timeout", state):
                await asyncio.sleep(1)  # exceed the 0.01s timeout

            # _deferred_mark_failed should have been scheduled
            mock_create_task.assert_called_once()
            coro = mock_create_task.call_args[0][0]
            # Clean up the coroutine to avoid warning
            coro.close()

    @pytest.mark.asyncio
    async def test_exception_writes_error_event_inline(self):
        server = _make_server()

        with patch(f"{MODULE}.get_messages", new_callable=AsyncMock, return_value=[
            (0, None, {"type": "response.created"}),
            (1, None, {"type": "response.output_text.delta"}),
        ]), \
             patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append, \
             patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update:
            state = {"seq": 2}
            async with server._task_scope("resp_err", state):
                raise ValueError("something broke")

            # Should have written error event at next seq (2)
            mock_append.assert_awaited_once()
            evt = mock_append.call_args.kwargs["stream_event"]
            assert evt["type"] == "error"
            assert evt["error"]["message"] == "something broke"
            assert evt["error"]["code"] == "task_failed"
            assert mock_append.call_args.args[1] == 2  # next_seq
            mock_update.assert_awaited_once_with("resp_err", "failed")

    @pytest.mark.asyncio
    async def test_exception_falls_back_to_deferred_on_db_failure(self):
        server = _make_server()

        with patch(f"{MODULE}.get_messages", new_callable=AsyncMock,
                   side_effect=RuntimeError("DB down")), \
             patch(f"{MODULE}.asyncio.create_task") as mock_create_task:
            state = {"seq": 0}
            async with server._task_scope("resp_fallback", state):
                raise ValueError("original error")

            # Inline cleanup failed → deferred task scheduled
            mock_create_task.assert_called_once()
            coro = mock_create_task.call_args[0][0]
            coro.close()


# ---------------------------------------------------------------------------
# P1: is_db_configured env var combinations
# ---------------------------------------------------------------------------


class TestIsDbConfigured:
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("LAKEBASE_INSTANCE_NAME", raising=False)
        monkeypatch.delenv("LAKEBASE_AUTOSCALING_ENDPOINT", raising=False)
        monkeypatch.delenv("LAKEBASE_AUTOSCALING_PROJECT", raising=False)
        monkeypatch.delenv("LAKEBASE_AUTOSCALING_BRANCH", raising=False)

    def test_no_vars(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        assert is_db_configured() is False

    def test_instance_name_only(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LAKEBASE_INSTANCE_NAME", "my-instance")
        assert is_db_configured() is True

    def test_empty_instance_name(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LAKEBASE_INSTANCE_NAME", "")
        assert is_db_configured() is False

    def test_autoscaling_both_set(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LAKEBASE_AUTOSCALING_PROJECT", "proj")
        monkeypatch.setenv("LAKEBASE_AUTOSCALING_BRANCH", "branch")
        assert is_db_configured() is True

    def test_autoscaling_only_project(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LAKEBASE_AUTOSCALING_PROJECT", "proj")
        assert is_db_configured() is False

    def test_autoscaling_only_branch(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LAKEBASE_AUTOSCALING_BRANCH", "branch")
        assert is_db_configured() is False

    def test_autoscaling_empty_strings(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LAKEBASE_AUTOSCALING_PROJECT", "")
        monkeypatch.setenv("LAKEBASE_AUTOSCALING_BRANCH", "branch")
        assert is_db_configured() is False

    def test_autoscaling_endpoint(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LAKEBASE_AUTOSCALING_ENDPOINT", "https://my-endpoint.com")
        assert is_db_configured() is True

    def test_autoscaling_endpoint_empty(self, monkeypatch):
        from databricks_ai_bridge.long_running.db import is_db_configured
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LAKEBASE_AUTOSCALING_ENDPOINT", "")
        assert is_db_configured() is False
