"""Tests for LongRunningAgentServer route registration, background handling, and SSE format."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from databricks_ai_bridge.long_running.server import (
    BACKGROUND_KEY,
    LongRunningAgentServer,
    _deferred_mark_failed,
    _sse_event,
)
from databricks_ai_bridge.long_running.settings import LongRunningSettings


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
        assert s.task_timeout_seconds == 1800.0
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
            return_value=("resp_123", "completed", time.time(), "trace_abc"),
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

        old_time = time.time() - 100  # well past timeout
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
            return_value=("resp_123", "in_progress", time.time(), None),
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
            return_value=("resp_123", "completed", time.time(), None),
        ), patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            return_value=[
                (0, None, {"type": "response.created", "id": "resp_123"}),
                (1, '{"text": "hi"}', {"type": "response.output_item.done", "item": {"text": "hi"}}),
            ],
        ):
            events = []
            async for chunk in server._stream_retrieve("resp_123", starting_after=-1):
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
            return_value=("resp_123", "failed", time.time(), None),
        ), patch(
            "databricks_ai_bridge.long_running.server.get_messages",
            new_callable=AsyncMock,
            return_value=[
                (0, None, {"type": "error", "error": {"message": "boom"}}),
            ],
        ):
            events = []
            async for chunk in server._stream_retrieve("resp_123", starting_after=-1):
                events.append(chunk)

            assert len(events) == 1
            assert "error" in events[0]
