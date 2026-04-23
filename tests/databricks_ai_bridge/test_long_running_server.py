"""Tests for LongRunningAgentServer route registration, background handling, and SSE format."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if __import__("sys").version_info < (3, 11):
    pytest.skip("long_running requires Python 3.11+", allow_module_level=True)
pytest.importorskip("fastapi")
pytest.importorskip("psycopg")

from databricks_ai_bridge.long_running.repository import ResponseInfo
from databricks_ai_bridge.long_running.server import (
    DEFAULT_SYNTHETIC_INTERRUPTED_OUTPUT,
    LongRunningAgentServer,
    _collect_prior_attempt_tool_events,
    _deferred_mark_failed,
    _inject_conversation_id,
    _rotate_conversation_id,
    _sanitize_request_input,
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


def _resp_info(
    response_id: str = "resp_123",
    status: str = "in_progress",
    created_at=None,
    trace_id: str | None = None,
    owner_pod_id: str | None = None,
    heartbeat_at=None,
    attempt_number: int = 1,
    original_request: dict | None = None,
) -> ResponseInfo:
    """Build a ResponseInfo with sensible defaults for tests.

    Mirrors the server's repository model so test setups stay terse even as
    durability columns grow over time.
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    return ResponseInfo(
        response_id=response_id,
        status=status,
        created_at=created_at,
        trace_id=trace_id,
        owner_pod_id=owner_pod_id,
        heartbeat_at=heartbeat_at,
        attempt_number=attempt_number,
        original_request=original_request,
    )


def _msg(seq: int, item=None, evt=None, attempt: int = 1):
    """Build a (seq, item, stream_event, attempt_number) tuple for get_messages mocks."""
    return (seq, item, evt, attempt)


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
        assert result.startswith("data: ")
        assert "event:" not in result
        assert result.endswith("\n\n")
        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["id"] == "resp_123"

    def test_string_data(self):
        result = _sse_event("error", "something went wrong")
        assert "event:" not in result
        assert result == "data: something went wrong\n\n"


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
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")
        event = {"type": "response.output_item.done", "item": {"id": "fake_id"}}
        result = server.transform_stream_event(event, "resp_real")
        assert result is event

    def test_subclass_override(self):
        class CustomServer(LongRunningAgentServer):
            def transform_stream_event(self, event, response_id):
                if isinstance(event, dict):
                    return {k: response_id if v == "FAKE" else v for k, v in event.items()}
                return event

        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
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
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")
        assert server.agent_type == "ResponsesAgent"

    def test_default_agent_type(self):
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer()
        assert server.agent_type == "ResponsesAgent"


class TestRouteRegistration:
    def test_routes_without_db(self):
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")

        routes = [r.path for r in server.app.routes if hasattr(r, "path")]
        # Parent routes should exist
        assert "/invocations" in routes
        # Both endpoints are always registered
        assert "/responses/{response_id}" in routes
        assert "/responses/{response_id}/cancel" in routes

    def test_routes_with_db(self):
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=True):
            server = LongRunningAgentServer("ResponsesAgent")

        routes = [r.path for r in server.app.routes if hasattr(r, "path")]
        assert "/responses/{response_id}" in routes
        assert "/responses/{response_id}/cancel" in routes


class TestCancelEndpoint:
    def test_cancel_returns_501(self):
        from starlette.testclient import TestClient

        server = _make_server()
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.post("/responses/resp_123/cancel")
        assert resp.status_code == 501
        assert "not yet implemented" in resp.json()["detail"].lower()


class TestRetrieveWithoutDb:
    def test_get_returns_501_without_db(self):
        from starlette.testclient import TestClient

        server = _make_server()
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.get("/responses/resp_123")
        assert resp.status_code == 501
        assert "database" in resp.json()["detail"].lower()


class TestStartingAfterValidation:
    def test_starting_after_without_stream_returns_400(self):
        from starlette.testclient import TestClient

        with patch(f"{MODULE}.is_db_configured", return_value=True):
            server = LongRunningAgentServer("ResponsesAgent")

        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.get("/responses/resp_123?starting_after=5&stream=false")
        assert resp.status_code == 400
        assert "starting_after" in resp.json()["detail"].lower()

    def test_starting_after_zero_without_stream_is_allowed(self):
        from starlette.testclient import TestClient

        with patch(f"{MODULE}.is_db_configured", return_value=True):
            server = LongRunningAgentServer("ResponsesAgent")

        with (
            patch(
                f"{MODULE}.get_response",
                new_callable=AsyncMock,
                return_value=_resp_info("resp_123", "in_progress"),
            ),
            patch(
                f"{MODULE}.get_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            client = TestClient(server.app, raise_server_exceptions=False)
            resp = client.get("/responses/resp_123?starting_after=0&stream=false")
            assert resp.status_code == 200


class TestDeferredMarkFailed:
    @pytest.mark.asyncio
    async def test_marks_response_failed(self):
        with (
            patch(
                "databricks_ai_bridge.long_running.server.get_messages",
                new_callable=AsyncMock,
                return_value=[_msg(0, None, {"type": "response.created"})],
            ) as mock_get,
            patch(
                "databricks_ai_bridge.long_running.server.get_response",
                new_callable=AsyncMock,
                return_value=_resp_info(),
            ),
            patch(
                "databricks_ai_bridge.long_running.server.append_message",
                new_callable=AsyncMock,
            ) as mock_append,
            patch(
                "databricks_ai_bridge.long_running.server.update_response_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
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
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")

        with patch(
            "databricks_ai_bridge.long_running.server.get_response",
            new_callable=AsyncMock,
            return_value=None,
        ):
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await server._handle_retrieve_request(
                    "resp_missing", stream=False, starting_after=0
                )
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_completed_returns_output(self):
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")

        with (
            patch(
                "databricks_ai_bridge.long_running.server.get_response",
                new_callable=AsyncMock,
                return_value=_resp_info("resp_123", "completed", trace_id="trace_abc"),
            ),
            patch(
                "databricks_ai_bridge.long_running.server.get_messages",
                new_callable=AsyncMock,
                return_value=[
                    _msg(
                        0,
                        '{"text": "hi"}',
                        {"type": "response.output_item.done", "item": {"text": "hi"}},
                    ),
                ],
            ),
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
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent", task_timeout_seconds=10.0)

        from datetime import timedelta

        old_time = datetime.now(timezone.utc) - timedelta(seconds=100)  # well past timeout
        with (
            patch(
                "databricks_ai_bridge.long_running.server.get_response",
                new_callable=AsyncMock,
                return_value=_resp_info("resp_stale", "in_progress", created_at=old_time),
            ),
            patch(
                "databricks_ai_bridge.long_running.server.get_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "databricks_ai_bridge.long_running.server.append_message",
                new_callable=AsyncMock,
            ) as mock_append,
            patch(
                "databricks_ai_bridge.long_running.server.update_response_status",
                new_callable=AsyncMock,
            ),
        ):
            result = await server._handle_retrieve_request(
                "resp_stale", stream=False, starting_after=0
            )
            assert result["status"] == "failed"
            mock_append.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_in_progress_returns_status(self):
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")

        with (
            patch(
                "databricks_ai_bridge.long_running.server.get_response",
                new_callable=AsyncMock,
                return_value=_resp_info("resp_123", "in_progress"),
            ),
            patch(
                "databricks_ai_bridge.long_running.server.get_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await server._handle_retrieve_request(
                "resp_123", stream=False, starting_after=0
            )
            assert result == {
                "id": "resp_123",
                "status": "in_progress",
                "attempt_number": 1,
            }


class TestStreamRetrieve:
    @pytest.mark.asyncio
    async def test_completed_stream(self):
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent", poll_interval_seconds=0.01)

        with (
            patch(
                "databricks_ai_bridge.long_running.server.get_response",
                new_callable=AsyncMock,
                return_value=_resp_info("resp_123", "completed"),
            ),
            patch(
                "databricks_ai_bridge.long_running.server.get_messages",
                new_callable=AsyncMock,
                return_value=[
                    _msg(0, None, {"type": "response.created", "id": "resp_123"}),
                    _msg(
                        1,
                        '{"text": "hi"}',
                        {"type": "response.output_item.done", "item": {"text": "hi"}},
                    ),
                ],
            ),
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
        with patch("databricks_ai_bridge.long_running.server.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent", poll_interval_seconds=0.01)

        with (
            patch(
                "databricks_ai_bridge.long_running.server.get_response",
                new_callable=AsyncMock,
                return_value=_resp_info("resp_123", "failed"),
            ),
            patch(
                "databricks_ai_bridge.long_running.server.get_messages",
                new_callable=AsyncMock,
                return_value=[
                    _msg(0, None, {"type": "error", "error": {"message": "boom"}}),
                ],
            ),
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

        with (
            patch(f"{MODULE}.get_stream_function", return_value=fake_stream),
            patch(f"{MODULE}.mlflow") as mock_mlflow,
            patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append,
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update,
            patch(f"{MODULE}.ResponsesAgent") as mock_ra,
        ):
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

        with (
            patch(f"{MODULE}.get_stream_function", return_value=fake_stream),
            patch(f"{MODULE}.mlflow") as mock_mlflow,
            patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append,
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock),
            patch(f"{MODULE}.ResponsesAgent") as mock_ra,
        ):
            mock_mlflow.start_span.return_value = span
            mock_ra.responses_agent_output_reducer.return_value = {"output": []}

            state = {"seq": 0}
            await server._do_background_stream("resp_t", {"input": "hi"}, False, state)

            assert len(transform_calls) == 2
            # Each call gets the correct response_id
            assert all(rid == "resp_t" for _, rid in transform_calls)
            # The transformed event is what gets persisted
            for call in mock_append.call_args_list:
                evt = (
                    call.kwargs.get("stream_event") or call.args[3]
                    if len(call.args) > 3
                    else call.kwargs.get("stream_event")
                )
                assert evt is not None
                assert evt.get("transformed") is True

    @pytest.mark.asyncio
    async def test_no_stream_fn_marks_failed(self):
        server = _make_server()

        with (
            patch(f"{MODULE}.get_stream_function", return_value=None),
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update,
        ):
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

        with (
            patch(f"{MODULE}.get_stream_function", return_value=fake_stream),
            patch(f"{MODULE}.mlflow") as mock_mlflow,
            patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append,
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock),
            patch(f"{MODULE}.ResponsesAgent") as mock_ra,
        ):
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

        with (
            patch(f"{MODULE}.get_invoke_function", return_value=fake_invoke),
            patch(f"{MODULE}.mlflow") as mock_mlflow,
            patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append,
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update,
            patch(f"{MODULE}.update_response_trace_id", new_callable=AsyncMock),
        ):
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

        with (
            patch(f"{MODULE}.get_invoke_function", return_value=fake_invoke),
            patch(f"{MODULE}.mlflow") as mock_mlflow,
            patch(f"{MODULE}.append_message", new_callable=AsyncMock),
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock),
            patch(f"{MODULE}.update_response_trace_id", new_callable=AsyncMock) as mock_trace,
        ):
            mock_mlflow.start_span.return_value = span

            state = {"seq": 0}
            await server._do_background_invoke("resp_inv", {"input": "hi"}, True, state)

            mock_trace.assert_awaited_once_with("resp_inv", "trace_abc123")

    @pytest.mark.asyncio
    async def test_no_invoke_fn_marks_failed(self):
        server = _make_server()

        with (
            patch(f"{MODULE}.get_invoke_function", return_value=None),
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update,
        ):
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

        with (
            patch(f"{MODULE}.get_invoke_function", return_value=sync_invoke),
            patch(f"{MODULE}.mlflow") as mock_mlflow,
            patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append,
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update,
            patch(f"{MODULE}.update_response_trace_id", new_callable=AsyncMock),
        ):
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

        with (
            patch(f"{MODULE}._deferred_mark_failed", new_callable=AsyncMock) as mock_deferred,
            patch(f"{MODULE}.asyncio.create_task") as mock_create_task,
        ):
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

        with (
            patch(
                f"{MODULE}.get_messages",
                new_callable=AsyncMock,
                return_value=[
                    _msg(0, None, {"type": "response.created"}),
                    _msg(1, None, {"type": "response.output_text.delta"}),
                ],
            ),
            patch(
                f"{MODULE}.get_response",
                new_callable=AsyncMock,
                return_value=_resp_info(),
            ),
            patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append,
            patch(f"{MODULE}.update_response_status", new_callable=AsyncMock) as mock_update,
        ):
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

        with (
            patch(
                f"{MODULE}.get_messages",
                new_callable=AsyncMock,
                side_effect=RuntimeError("DB down"),
            ),
            patch(f"{MODULE}.asyncio.create_task") as mock_create_task,
        ):
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


class TestConstructorParams:
    def test_stores_db_autoscaling_endpoint(self):
        server = _make_server(db_autoscaling_endpoint="ep-abc")
        assert server._db_autoscaling_endpoint == "ep-abc"

    def test_stores_all_db_params(self):
        server = _make_server(
            db_instance_name="inst",
            db_autoscaling_endpoint="ep",
            db_project="proj",
            db_branch="br",
        )
        assert server._db_instance_name == "inst"
        assert server._db_autoscaling_endpoint == "ep"
        assert server._db_project == "proj"
        assert server._db_branch == "br"

    def test_db_params_default_to_none(self):
        server = _make_server()
        assert server._db_instance_name is None
        assert server._db_autoscaling_endpoint is None
        assert server._db_project is None
        assert server._db_branch is None


class TestLifespanPlumbing:
    @pytest.mark.asyncio
    async def test_lifespan_calls_init_db_with_all_params(self):
        with patch(f"{MODULE}.is_db_configured", return_value=True):
            server = LongRunningAgentServer(
                "ResponsesAgent",
                db_instance_name="inst",
                db_autoscaling_endpoint="ep",
                db_project="proj",
                db_branch="br",
                db_statement_timeout_ms=3000,
                cleanup_timeout_seconds=6.0,
            )

        lifespan = server.app.router.lifespan_context
        with (
            patch(f"{MODULE}.init_db", new_callable=AsyncMock) as mock_init,
            patch(f"{MODULE}.dispose_db", new_callable=AsyncMock) as mock_dispose,
        ):
            async with lifespan(MagicMock()):
                mock_init.assert_awaited_once_with(
                    instance_name="inst",
                    autoscaling_endpoint="ep",
                    project="proj",
                    branch="br",
                    db_statement_timeout_ms=3000,
                )
            mock_dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_with_endpoint_only(self):
        with patch(f"{MODULE}.is_db_configured", return_value=True):
            server = LongRunningAgentServer(
                "ResponsesAgent",
                db_autoscaling_endpoint="ep-only",
            )

        lifespan = server.app.router.lifespan_context
        with (
            patch(f"{MODULE}.init_db", new_callable=AsyncMock) as mock_init,
            patch(f"{MODULE}.dispose_db", new_callable=AsyncMock),
        ):
            async with lifespan(MagicMock()):
                mock_init.assert_awaited_once_with(
                    instance_name=None,
                    autoscaling_endpoint="ep-only",
                    project=None,
                    branch=None,
                    db_statement_timeout_ms=5000,
                )

    @pytest.mark.asyncio
    async def test_lifespan_not_set_when_db_not_configured(self):
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")

        with patch(f"{MODULE}.init_db", new_callable=AsyncMock) as mock_init:
            # The lifespan should be the default (not our custom _db_lifespan),
            # so init_db should never be called.
            # The retrieve route IS registered but returns 501 without DB.
            routes = [r.path for r in server.app.routes if hasattr(r, "path")]
            assert "/responses/{response_id}" in routes
            mock_init.assert_not_awaited()


# ---------------------------------------------------------------------------
# Durable resume: claim/heartbeat/attempt_number/sentinel
# ---------------------------------------------------------------------------


class TestSanitizeRequestInput:
    """Full-history orphan walker — catches mid-history orphans that neither
    the LangGraph middleware (trailing-only) nor session.repair() (session-only)
    cover. See rotation-findings.md Test E."""

    def test_empty_input_is_noop(self):
        assert _sanitize_request_input({}) == {}
        assert _sanitize_request_input({"input": []}) == {"input": []}

    def test_passes_through_paired_call_and_output(self):
        items = [
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        out = _sanitize_request_input({"input": list(items)})
        assert out["input"] == items

    def test_injects_synthetic_output_for_trailing_orphan_call(self):
        items = [
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        ]
        out = _sanitize_request_input({"input": items})
        assert len(out["input"]) == 3
        assert out["input"][2] == {
            "type": "function_call_output",
            "call_id": "c1",
            "output": DEFAULT_SYNTHETIC_INTERRUPTED_OUTPUT,
        }

    def test_injects_synthetic_output_for_midhistory_orphan_call(self):
        # The case that today's middleware misses (Test E).
        items = [
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"role": "user", "content": "different question"},
        ]
        out = _sanitize_request_input({"input": items})
        assert len(out["input"]) == 4
        assert out["input"][1]["type"] == "function_call"
        assert out["input"][2] == {
            "type": "function_call_output",
            "call_id": "c1",
            "output": DEFAULT_SYNTHETIC_INTERRUPTED_OUTPUT,
        }
        assert out["input"][3] == {"role": "user", "content": "different question"}

    def test_drops_orphan_output_with_no_matching_call(self):
        # The LangGraph stream-event attempt-boundary artifact.
        items = [
            {"role": "user", "content": "hi"},
            {"type": "function_call_output", "call_id": "c-ghost", "output": "x"},
            {"role": "user", "content": "follow-up"},
        ]
        out = _sanitize_request_input({"input": items})
        assert out["input"] == [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "follow-up"},
        ]

    def test_dedupes_duplicate_calls_and_outputs(self):
        items = [
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]
        out = _sanitize_request_input({"input": items})
        assert len(out["input"]) == 2
        assert out["input"][0]["type"] == "function_call"
        assert out["input"][1]["type"] == "function_call_output"

    def test_recognizes_chat_completions_shape_as_declaring_call_id(self):
        # An assistant message with tool_calls counts as "declaring" a call_id,
        # so a matching function_call_output further down is NOT dropped.
        items = [
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {"id": "tc-1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"type": "function_call_output", "call_id": "tc-1", "output": "ok"},
        ]
        out = _sanitize_request_input({"input": list(items)})
        assert out["input"] == items

    def test_non_dict_items_pass_through(self):
        items = [{"role": "user", "content": "hi"}, "not-a-dict", 42]
        out = _sanitize_request_input({"input": list(items)})
        assert out["input"] == items


class TestCollectPriorAttemptToolEvents:
    """Gather function_call / function_call_output items emitted during a
    prior attempt so the next attempt can inherit already-completed tool
    results instead of re-running them from scratch."""

    def _event(self, seq, attempt, item_type, call_id, output=None):
        item = {"type": item_type, "call_id": call_id, "name": "f", "arguments": "{}"}
        if output is not None:
            item = {"type": item_type, "call_id": call_id, "output": output}
        evt = {"type": "response.output_item.done", "item": item}
        return (seq, None, evt, attempt)

    def test_filters_to_requested_prior_attempt(self):
        messages = [
            self._event(0, 1, "function_call", "c1"),
            self._event(1, 1, "function_call_output", "c1", output="ok"),
            # attempt 2's events should not be returned when asking for attempt 1.
            self._event(2, 2, "function_call", "c2"),
        ]
        out = _collect_prior_attempt_tool_events(messages, prior_attempt_number=1)
        assert [i["call_id"] for i in out] == ["c1", "c1"]
        assert [i["type"] for i in out] == ["function_call", "function_call_output"]

    def test_only_output_item_done_events_count(self):
        noise = (
            0,
            None,
            {"type": "response.output_text.delta", "delta": "hi"},
            1,
        )
        messages = [noise, self._event(1, 1, "function_call", "c1")]
        out = _collect_prior_attempt_tool_events(messages, prior_attempt_number=1)
        assert len(out) == 1
        assert out[0]["call_id"] == "c1"

    def test_completed_assistant_message_items_are_skipped(self):
        # Narrative `message` items Claude interleaves between tool calls
        # would break Anthropic's "tool_use immediately followed by
        # tool_result" rule if inherited. Only function_call/output pairs
        # flow through; completed messages are dropped.
        messages = [
            (
                0,
                None,
                {
                    "type": "response.output_item.done",
                    "item": {"type": "message", "role": "assistant", "content": "Let me check"},
                },
                1,
            ),
            self._event(1, 1, "function_call", "c1"),
            self._event(2, 1, "function_call_output", "c1", output="ok"),
        ]
        out = _collect_prior_attempt_tool_events(messages, prior_attempt_number=1)
        # Only the function_call + function_call_output pair — the message
        # is intentionally skipped.
        assert len(out) == 2
        assert out[0]["type"] == "function_call"
        assert out[1]["type"] == "function_call_output"

    def test_reassembles_partial_text_from_delta_events(self):
        # Attempt crashed mid-stream: item.added + deltas but no item.done.
        # The collector should synthesize a message item from accumulated deltas
        # so the next attempt sees where narration trailed off.
        messages = [
            (
                0,
                None,
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "msg_1"},
                },
                1,
            ),
            (
                1,
                None,
                {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello, "},
                1,
            ),
            (
                2,
                None,
                {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "world"},
                1,
            ),
            # No item.done — crash.
        ]
        out = _collect_prior_attempt_tool_events(messages, prior_attempt_number=1)
        assert len(out) == 1
        assert out[0]["type"] == "message"
        assert out[0]["role"] == "assistant"
        assert out[0]["content"][0]["text"] == "Hello, world"

    def test_ignores_partial_text_if_item_eventually_completed(self):
        # Deltas streamed, then item.done landed — since completed message
        # items are no longer inherited at all, and the partial reassembly
        # only fires when .done is missing, this case produces an empty
        # inherited list.
        messages = [
            (
                0,
                None,
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "msg_1"},
                },
                1,
            ),
            (
                1,
                None,
                {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello"},
                1,
            ),
            (
                2,
                None,
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello, world"}],
                    },
                },
                1,
            ),
        ]
        out = _collect_prior_attempt_tool_events(messages, prior_attempt_number=1)
        # Completed messages are intentionally NOT inherited, and partial
        # reassembly cleared its tracker when .done arrived → empty list.
        assert out == []

    def test_skips_unknown_item_types(self):
        # Item types outside the allow-list (e.g., future event kinds like
        # file_search_call / code_interpreter_call) are dropped — safer than
        # forwarding them to the handler without review.
        messages = [
            (
                0,
                None,
                {
                    "type": "response.output_item.done",
                    "item": {"type": "file_search_call", "results": []},
                },
                1,
            )
        ]
        out = _collect_prior_attempt_tool_events(messages, prior_attempt_number=1)
        assert out == []


class TestRotateConversationId:
    def test_rotate_drops_thread_id_and_sets_rotated_context(self):
        r = {"custom_inputs": {"thread_id": "t1", "user_id": "u"}, "context": {}}
        out = _rotate_conversation_id(r, new_attempt_number=2, response_id="resp_x")
        assert "thread_id" not in out["custom_inputs"]
        assert out["custom_inputs"]["user_id"] == "u"
        assert out["context"]["conversation_id"] == "t1::attempt-2"

    def test_rotate_drops_session_id(self):
        r = {"custom_inputs": {"session_id": "s1"}, "context": {}}
        out = _rotate_conversation_id(r, new_attempt_number=2, response_id="resp_x")
        assert "session_id" not in out["custom_inputs"]
        assert out["context"]["conversation_id"] == "s1::attempt-2"

    def test_rotate_falls_back_to_context_conversation_id(self):
        r = {"custom_inputs": {}, "context": {"conversation_id": "c-abc"}}
        out = _rotate_conversation_id(r, new_attempt_number=3, response_id="resp_x")
        assert out["context"]["conversation_id"] == "c-abc::attempt-3"

    def test_rotate_falls_back_to_response_id_as_last_resort(self):
        r = {"custom_inputs": {}, "context": {}}
        out = _rotate_conversation_id(r, new_attempt_number=2, response_id="resp_x")
        assert out["context"]["conversation_id"] == "resp_x::attempt-2"

    def test_rotate_handles_missing_custom_inputs_key(self):
        r = {"context": {"conversation_id": "c-abc"}}
        out = _rotate_conversation_id(r, new_attempt_number=2, response_id="resp_x")
        assert out["context"]["conversation_id"] == "c-abc::attempt-2"
        assert out["custom_inputs"] == {}


class TestInjectConversationId:
    """Anchoring an otherwise-anonymous request to a response_id guarantees a
    resumed run on a new pod resolves to the same agent-SDK thread/session."""

    def test_injects_when_nothing_set(self):
        r = {"input": [], "custom_inputs": {}, "context": {}}
        out = _inject_conversation_id(r, "resp_abc")
        assert out["context"]["conversation_id"] == "resp_abc"

    def test_respects_existing_conversation_id(self):
        r = {"input": [], "context": {"conversation_id": "user-set"}}
        out = _inject_conversation_id(r, "resp_abc")
        assert out["context"]["conversation_id"] == "user-set"

    def test_respects_thread_id_from_custom_inputs(self):
        r = {"input": [], "custom_inputs": {"thread_id": "t-1"}, "context": {}}
        out = _inject_conversation_id(r, "resp_abc")
        # When the client already pinned a thread, we don't overwrite — the
        # template's _get_or_create_thread_id picks up custom_inputs first.
        assert "conversation_id" not in (out["context"] or {})

    def test_respects_session_id_from_custom_inputs(self):
        r = {"input": [], "custom_inputs": {"session_id": "s-1"}, "context": {}}
        out = _inject_conversation_id(r, "resp_abc")
        assert "conversation_id" not in (out["context"] or {})

    def test_handles_missing_context_key(self):
        r = {"input": [], "custom_inputs": {}}
        out = _inject_conversation_id(r, "resp_abc")
        assert out["context"]["conversation_id"] == "resp_abc"

    def test_does_not_mutate_input(self):
        r = {"input": [], "custom_inputs": {}, "context": {}}
        _inject_conversation_id(r, "resp_abc")
        assert r["context"] == {}  # original untouched


class TestHandleBackgroundRequestPersistsDurabilityState:
    """Background request entry point should now stamp the response row with
    the caller's pod, the original request body, and a conversation anchor."""

    @pytest.mark.asyncio
    async def test_persists_owner_and_original_request(self):
        with patch(f"{MODULE}.is_db_configured", return_value=True):
            server = LongRunningAgentServer("ResponsesAgent")
        _mock_validator(server)

        captured: dict = {}

        async def fake_create_response(
            response_id, status, *, owner_pod_id=None, original_request=None
        ):
            captured["response_id"] = response_id
            captured["status"] = status
            captured["owner_pod_id"] = owner_pod_id
            captured["original_request"] = original_request

        with (
            patch(f"{MODULE}.create_response", side_effect=fake_create_response),
            patch("asyncio.create_task") as mock_create_task,
        ):
            result = await server._handle_background_request(
                {"input": [{"role": "user", "content": "hi"}]},
                is_streaming=False,
                return_trace_id=False,
            )

        assert captured["status"] == "in_progress"
        assert captured["owner_pod_id"]  # non-empty
        # original_request should include input + injected conversation_id.
        orig = captured["original_request"]
        assert orig["input"] == [{"role": "user", "content": "hi"}]
        assert orig["context"]["conversation_id"] == captured["response_id"]
        # Return shape: immediate response_obj, not a stream.
        assert result["id"] == captured["response_id"]
        assert result["status"] == "in_progress"
        mock_create_task.assert_called_once()


class TestTryClaimAndResume:
    @pytest.mark.asyncio
    async def test_no_op_when_completed(self):
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")
        resp = _resp_info(status="completed")
        with patch(f"{MODULE}.claim_stale_response", new_callable=AsyncMock) as mock_claim:
            result = await server._try_claim_and_resume("resp_x", resp)
        assert result is None
        mock_claim.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_grace_period_for_fresh_run(self):
        """Just-started runs get a grace window before they're claim-eligible."""
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer(
                "ResponsesAgent", heartbeat_stale_threshold_seconds=15.0
            )
        # created 2s ago, no heartbeat yet → should NOT be claimed.
        from datetime import timedelta

        resp = _resp_info(
            status="in_progress",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=2),
            heartbeat_at=None,
            original_request={"input": []},
        )
        with patch(f"{MODULE}.claim_stale_response", new_callable=AsyncMock) as mock_claim:
            result = await server._try_claim_and_resume("resp_x", resp)
        assert result is None
        mock_claim.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_op_without_original_request(self):
        """Legacy rows created before durability metadata can't be resumed."""
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")
        from datetime import timedelta

        resp = _resp_info(
            status="in_progress",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=300),
            heartbeat_at=None,
            original_request=None,
        )
        with patch(f"{MODULE}.claim_stale_response", new_callable=AsyncMock) as mock_claim:
            result = await server._try_claim_and_resume("resp_x", resp)
        assert result is None
        mock_claim.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_claim_fails_returns_none(self):
        """Another pod won the race — we quietly step aside."""
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")
        from datetime import timedelta

        resp = _resp_info(
            status="in_progress",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=300),
            heartbeat_at=datetime.now(timezone.utc) - timedelta(seconds=300),
            original_request={"input": [{"role": "user"}]},
        )
        with (
            patch(
                f"{MODULE}.claim_stale_response", new_callable=AsyncMock, return_value=None
            ) as mock_claim,
            patch(f"{MODULE}.append_message", new_callable=AsyncMock) as mock_append,
        ):
            result = await server._try_claim_and_resume("resp_x", resp)
        assert result is None
        mock_claim.assert_awaited_once()
        mock_append.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_successful_claim_spawns_resume_and_emits_sentinel(self):
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")
        from datetime import timedelta

        resp = _resp_info(
            status="in_progress",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=300),
            heartbeat_at=datetime.now(timezone.utc) - timedelta(seconds=100),
            original_request={
                "input": [{"role": "user", "content": "hi"}],
                "custom_inputs": {"user_id": "u"},
                "context": {"conversation_id": "resp_x"},
            },
        )
        captured: dict = {}

        async def fake_append(response_id, seq, *, item=None, stream_event=None, attempt_number=1):
            captured["seq"] = seq
            captured["event"] = stream_event
            captured["attempt_tag"] = attempt_number

        with (
            patch(f"{MODULE}.claim_stale_response", new_callable=AsyncMock, return_value=2),
            patch(
                f"{MODULE}.get_messages",
                new_callable=AsyncMock,
                return_value=[_msg(0, None, {}), _msg(1, None, {})],
            ),
            patch(f"{MODULE}.append_message", side_effect=fake_append),
            patch("asyncio.create_task") as mock_create_task,
        ):
            attempt = await server._try_claim_and_resume("resp_x", resp)

        assert attempt == 2
        # Sentinel is written at next_seq (existing seqs were 0 and 1).
        assert captured["seq"] == 2
        assert captured["event"]["type"] == "response.resumed"
        assert captured["event"]["attempt"] == 2
        assert captured["attempt_tag"] == 2
        # A resume task is spawned; it was not awaited synchronously.
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_replays_input_and_rotates_conversation_id(self):
        """Resume must replay original_request.input (not blank it) and rotate
        the conversation anchor so the handler resolves to a fresh thread /
        session for the new attempt. Prevents the LangGraph stream-event
        attempt-boundary orphan artifact (rotation-findings.md)."""
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")
        from datetime import timedelta

        resp = _resp_info(
            status="in_progress",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=300),
            heartbeat_at=datetime.now(timezone.utc) - timedelta(seconds=100),
            original_request={
                "input": [{"role": "user", "content": "hi"}],
                "custom_inputs": {"thread_id": "t1", "user_id": "u"},
                "context": {},
            },
        )

        captured_tasks = []

        def capture_task(coro, *, name=None):
            captured_tasks.append((coro, name))

            class _Fake:
                def cancel(self):
                    pass

                def add_done_callback(self, cb):
                    pass

            return _Fake()

        with (
            patch(f"{MODULE}.claim_stale_response", new_callable=AsyncMock, return_value=2),
            patch(f"{MODULE}.get_messages", new_callable=AsyncMock, return_value=[]),
            patch(f"{MODULE}.append_message", new_callable=AsyncMock),
            patch("asyncio.create_task", side_effect=capture_task),
            patch.object(server, "_run_background_stream", new_callable=AsyncMock) as mock_run,
        ):
            await server._try_claim_and_resume("resp_x", resp)

        assert len(captured_tasks) == 1
        coro, _name = captured_tasks[0]
        await coro
        mock_run.assert_awaited_once()
        args, kwargs = mock_run.call_args
        resume_request = args[1] if len(args) > 1 else kwargs["request_data"]
        dumped = (
            resume_request.model_dump() if hasattr(resume_request, "model_dump") else resume_request
        )
        # Input is REPLAYED (not blanked) — the LLM sees full pre-crash history.
        # The MLflow validator normalizes the shape (adds "type": "message" etc.)
        # so compare essentials.
        assert len(dumped["input"]) == 1
        assert dumped["input"][0]["role"] == "user"
        assert dumped["input"][0]["content"] == "hi"
        # thread_id was dropped so the handler's priority-2 fallback wins.
        assert "thread_id" not in (dumped["custom_inputs"] or {})
        # Other custom_inputs keys are preserved.
        assert dumped["custom_inputs"]["user_id"] == "u"
        # conversation_id is rotated to a per-attempt value anchored on t1.
        assert dumped["context"]["conversation_id"] == "t1::attempt-2"
        assert kwargs.get("attempt_number") == 2

    @pytest.mark.asyncio
    async def test_resume_rotation_anchors_on_context_conversation_id(self):
        """When the client didn't pin a thread_id/session_id, rotation uses
        the injected context.conversation_id as the base anchor."""
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")
        from datetime import timedelta

        resp = _resp_info(
            status="in_progress",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=300),
            heartbeat_at=datetime.now(timezone.utc) - timedelta(seconds=100),
            original_request={
                "input": [{"role": "user", "content": "hi"}],
                "custom_inputs": {},
                "context": {"conversation_id": "resp_x"},
            },
        )

        captured_tasks = []

        def capture_task(coro, *, name=None):
            captured_tasks.append((coro, name))

            class _Fake:
                def cancel(self):
                    pass

                def add_done_callback(self, cb):
                    pass

            return _Fake()

        with (
            patch(f"{MODULE}.claim_stale_response", new_callable=AsyncMock, return_value=3),
            patch(f"{MODULE}.get_messages", new_callable=AsyncMock, return_value=[]),
            patch(f"{MODULE}.append_message", new_callable=AsyncMock),
            patch("asyncio.create_task", side_effect=capture_task),
            patch.object(server, "_run_background_stream", new_callable=AsyncMock) as mock_run,
        ):
            await server._try_claim_and_resume("resp_x", resp)

        assert len(captured_tasks) == 1
        coro, _name = captured_tasks[0]
        await coro
        mock_run.assert_awaited_once()
        args, kwargs = mock_run.call_args
        resume_request = args[1] if len(args) > 1 else kwargs["request_data"]
        dumped = (
            resume_request.model_dump() if hasattr(resume_request, "model_dump") else resume_request
        )
        # Rotation anchors on the stored context.conversation_id (priority 2).
        # Note: re-rotating in a subsequent attempt would re-anchor on the
        # ORIGINAL stored value, not the previous rotation — no stacking.
        assert dumped["context"]["conversation_id"] == "resp_x::attempt-3"


class TestRetrieveTriggersLazyClaim:
    @pytest.mark.asyncio
    async def test_retrieve_calls_try_claim(self):
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer("ResponsesAgent")

        resp = _resp_info("resp_x", "in_progress")
        with (
            patch(f"{MODULE}.get_response", new_callable=AsyncMock, return_value=resp),
            patch(f"{MODULE}.get_messages", new_callable=AsyncMock, return_value=[]),
            patch.object(
                server, "_try_claim_and_resume", new_callable=AsyncMock, return_value=None
            ) as mock_claim,
        ):
            await server._handle_retrieve_request("resp_x", stream=False, starting_after=0)

        mock_claim.assert_awaited_once()


class TestHeartbeatContextManager:
    @pytest.mark.asyncio
    async def test_writes_heartbeat_periodically(self):
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer(
                "ResponsesAgent",
                heartbeat_interval_seconds=0.05,
                heartbeat_stale_threshold_seconds=1.0,
            )

        with patch(f"{MODULE}.heartbeat_response", new_callable=AsyncMock) as mock_hb:
            async with server._heartbeat("resp_x"):
                await asyncio.sleep(0.2)  # enough time for 2+ heartbeats

        # Heartbeat interval is 0.05s so we should see at least 2 writes.
        assert mock_hb.await_count >= 2
        for call in mock_hb.await_args_list:
            assert call.args[0] == "resp_x"

    @pytest.mark.asyncio
    async def test_stops_cleanly_on_exit(self):
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer(
                "ResponsesAgent",
                heartbeat_interval_seconds=0.05,
                heartbeat_stale_threshold_seconds=1.0,
            )

        with patch(f"{MODULE}.heartbeat_response", new_callable=AsyncMock) as mock_hb:
            async with server._heartbeat("resp_x"):
                pass  # immediate exit

            # Give the heartbeat loop a chance to observe the stop signal.
            await asyncio.sleep(0.1)
            writes_after_exit = mock_hb.await_count

            await asyncio.sleep(0.15)
            # No new writes after the scope closed.
            assert mock_hb.await_count == writes_after_exit

    @pytest.mark.asyncio
    async def test_db_error_does_not_interrupt_body(self):
        """Heartbeat failures are logged, not raised — the stale check catches
        real death, so a transient write miss must not kill a live run."""
        with patch(f"{MODULE}.is_db_configured", return_value=False):
            server = LongRunningAgentServer(
                "ResponsesAgent",
                heartbeat_interval_seconds=0.05,
                heartbeat_stale_threshold_seconds=1.0,
            )

        body_ran = False
        with patch(
            f"{MODULE}.heartbeat_response",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            async with server._heartbeat("resp_x"):
                await asyncio.sleep(0.1)
                body_ran = True
        assert body_ran


class TestSettingsHeartbeatValidation:
    def test_stale_must_exceed_interval(self):
        with pytest.raises(ValueError, match="heartbeat_stale_threshold_seconds"):
            LongRunningSettings(
                heartbeat_interval_seconds=5.0,
                heartbeat_stale_threshold_seconds=5.0,
            )

    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="heartbeat_interval_seconds must be positive"):
            LongRunningSettings(heartbeat_interval_seconds=0)

    def test_defaults_match_chat_ux(self):
        # 3s interval + 15s stale gives ~5 heartbeats before a pod is considered
        # dead — snug enough to recover conversations within a user's
        # "reconnecting..." patience window.
        s = LongRunningSettings()
        assert s.heartbeat_interval_seconds == 3.0
        assert s.heartbeat_stale_threshold_seconds == 10.0


class TestDebugKillTask:
    """The opt-in debug-kill endpoint lets integration tests simulate a crash
    against a deployed pod without restarting the whole app. Off by default
    because exposing task cancellation bypasses the normal cleanup path."""

    def test_endpoint_absent_by_default(self):
        from starlette.testclient import TestClient

        with patch(f"{MODULE}.is_db_configured", return_value=True):
            server = LongRunningAgentServer("ResponsesAgent")
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.post("/_debug/kill_task/resp_x")
        assert resp.status_code == 404  # route not registered

    def test_endpoint_registered_when_env_set(self, monkeypatch):
        from starlette.testclient import TestClient

        monkeypatch.setenv("LONG_RUNNING_ENABLE_DEBUG_KILL", "1")
        with patch(f"{MODULE}.is_db_configured", return_value=True):
            server = LongRunningAgentServer("ResponsesAgent")
        client = TestClient(server.app, raise_server_exceptions=False)
        # No in-flight task for this response_id on this pod → 404, not 405.
        resp = client.post("/_debug/kill_task/resp_missing")
        assert resp.status_code == 404
        assert "No in-flight task" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_cancels_tracked_task(self, monkeypatch):
        """Direct-call variant: skip the TestClient (which is sync and blocks
        the loop) and call the handler logic through _running_tasks directly.
        Covers the important behavior: cancelling a tracked task propagates
        CancelledError and the tracking dict is cleared by the done-callback.
        """
        monkeypatch.setenv("LONG_RUNNING_ENABLE_DEBUG_KILL", "1")
        with patch(f"{MODULE}.is_db_configured", return_value=True):
            server = LongRunningAgentServer("ResponsesAgent")

        cancel_event = asyncio.Event()

        async def long_running():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancel_event.set()
                raise

        task = asyncio.create_task(long_running())
        server._track_task("resp_tracked", task)

        # Yield once so the new task can start waiting on sleep(60).
        await asyncio.sleep(0)
        assert "resp_tracked" in server._running_tasks

        task.cancel()
        # Expect CancelledError from awaiting the task itself, and the cancel
        # event set inside the except handler before the re-raise.
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancel_event.is_set()
        # done-callback (scheduled on loop) clears the registration after the
        # task completes — give it one more tick.
        await asyncio.sleep(0)
        assert "resp_tracked" not in server._running_tasks
