"""Long-running agent server with Lakebase persistence and background mode."""

import asyncio
import inspect
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import mlflow
from fastapi import HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from mlflow.genai.agent_server import get_invoke_function, get_stream_function
from mlflow.genai.agent_server.server import (
    RETURN_TRACE_HEADER,
    AgentServer,
)
from mlflow.genai.agent_server.server import (
    STREAM_KEY as MLFLOW_STREAM_KEY,
)
from mlflow.genai.agent_server.utils import get_request_headers, set_request_headers
from mlflow.pyfunc import ResponsesAgent
from mlflow.tracing.constant import SpanAttributeKey

from databricks_ai_bridge.long_running.db import dispose_db, init_db, is_db_configured
from databricks_ai_bridge.long_running.repository import (
    append_message,
    create_response,
    get_messages,
    get_response,
    update_response_status,
    update_response_trace_id,
)
from databricks_ai_bridge.long_running.settings import LongRunningSettings
from databricks_ai_bridge.utils.annotations import experimental

logger = logging.getLogger(__name__)

BACKGROUND_KEY = "background"


async def _deferred_mark_failed(
    response_id: str, delay: float = 2.0, reason: str = "Task timed out"
) -> None:
    """Mark a response as failed after a short delay.

    Runs as an independent asyncio task so the caller (``_task_scope``) can
    return immediately.  The delay lets the connection pool stabilise after
    a cancellation before we attempt new DB writes.
    """
    try:
        await asyncio.sleep(delay)

        # TODO: sequence number computation is racy under concurrent writers.
        # Acceptable at current scale; for high-QPS use a DB-assigned sequence
        # or SELECT FOR UPDATE on the response row to serialise writers.
        async with asyncio.timeout(delay):
            existing = await get_messages(response_id, after_sequence=None)
            next_seq = max((seq for seq, _, _ in existing), default=-1) + 1

            error_event = {
                "type": "error",
                "error": {
                    "message": reason,
                    "type": "server_error",
                    "code": "task_timeout",
                },
            }
            await append_message(response_id, next_seq, item=None, stream_event=error_event)
            await update_response_status(response_id, "failed")

        logger.info("Marked %s as failed (reason: %s)", response_id, reason)
    except TimeoutError:
        logger.error(
            "Timed out marking %s as failed; stale-run check will catch it",
            response_id,
        )
    except Exception:
        logger.exception(
            "Failed to mark %s as failed; stale-run check will catch it",
            response_id,
        )


def _sse_event(event_type: str, data: dict[str, Any] | str) -> str:
    """Format an SSE event per Open Responses spec."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event_type}\ndata: {payload}\n\n"


def _age_seconds(created_at: datetime) -> float:
    """Return the age of a timestamp in seconds."""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (now - created_at).total_seconds()


@experimental
class LongRunningAgentServer(AgentServer):
    """AgentServer subclass adding background mode and retrieve endpoints.

    Only compatible with ``ResponsesAgent`` mode.

    Args:
        enable_chat_proxy: Whether to enable the chat proxy endpoint.
        db_instance_name: Lakebase provisioned instance name.
        db_autoscaling_endpoint: Lakebase autoscaling endpoint URL.
        db_project: Lakebase autoscaling project.
        db_branch: Lakebase autoscaling branch.
        task_timeout_seconds: Max time for a background task before timeout.
        poll_interval_seconds: Interval between DB polls when streaming.
        db_statement_timeout_ms: Postgres statement timeout.
        cleanup_timeout_seconds: Timeout for DB cleanup after task failure.
    """

    _SUPPORTED_AGENT_TYPE = "ResponsesAgent"

    def __init__(
        self,
        agent_type=_SUPPORTED_AGENT_TYPE,
        *,
        enable_chat_proxy=False,
        # DB config
        db_instance_name: str | None = None,
        db_autoscaling_endpoint: str | None = None,
        db_project: str | None = None,
        db_branch: str | None = None,
        # Settings (override defaults)
        task_timeout_seconds: float = 3600.0,
        poll_interval_seconds: float = 1.0,
        db_statement_timeout_ms: int = 5000,
        cleanup_timeout_seconds: float = 7.0,
    ):
        if agent_type != self._SUPPORTED_AGENT_TYPE:
            raise ValueError(
                f"LongRunningAgentServer only supports '{self._SUPPORTED_AGENT_TYPE}', "
                f"got '{agent_type}'"
            )
        self._settings = LongRunningSettings(
            task_timeout_seconds=task_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            db_statement_timeout_ms=db_statement_timeout_ms,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
        )
        self._db_instance_name = db_instance_name
        self._db_autoscaling_endpoint = db_autoscaling_endpoint
        self._db_project = db_project
        self._db_branch = db_branch
        super().__init__(agent_type, enable_chat_proxy=enable_chat_proxy)

    def _setup_routes(self) -> None:
        """Register routes. Reuses parent's POST /invocations and POST /responses.

        Adds GET /responses/{id} for polling/streaming when DB is configured.
        Auto-registers startup/shutdown events for DB lifecycle.
        """
        super()._setup_routes()

        if not is_db_configured():
            logger.warning("Database not configured. Background mode disabled.")
            return

        @asynccontextmanager
        async def _db_lifespan(app):
            await init_db(
                instance_name=self._db_instance_name,
                autoscaling_endpoint=self._db_autoscaling_endpoint,
                project=self._db_project,
                branch=self._db_branch,
                db_statement_timeout_ms=self._settings.db_statement_timeout_ms,
            )
            yield
            await dispose_db()

        self.app.router.lifespan_context = _db_lifespan

        @self.app.get("/responses/{response_id}")
        async def retrieve_endpoint(
            response_id: str,
            stream: bool = Query(False, description="Stream results as SSE"),
            starting_after: int = Query(
                0, ge=0, description="Resume from sequence number (0 means fetch all)"
            ),
        ):
            return await self._handle_retrieve_request(
                response_id,
                stream=stream,
                starting_after=starting_after,
            )

    async def _handle_invocations_request(
        self, request: Request
    ) -> dict[str, Any] | StreamingResponse:
        """Handle POST /responses and POST /invocations.

        Intentionally overrides the parent implementation to add background
        mode support. Non-background requests delegate to the same
        ``_handle_stream_request`` / ``_handle_invoke_request`` helpers.

        When background=true and DB is configured, returns a response_id
        immediately and starts the agent loop in the background.
        """
        set_request_headers(dict(request.headers))

        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON in request body: {e!s}"
            ) from None

        is_background = data.get(BACKGROUND_KEY, False)
        is_streaming = data.get(MLFLOW_STREAM_KEY, False)
        data = {k: v for k, v in data.items() if k not in (BACKGROUND_KEY, MLFLOW_STREAM_KEY)}
        return_trace_id = (
            (get_request_headers().get(RETURN_TRACE_HEADER) or "").lower() == "true"
        )

        try:
            request_data = self.validator.validate_and_convert_request(data)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters for {self.agent_type}: {e}",
            ) from None

        if is_background and is_db_configured():
            return await self._handle_background_request(
                request_data, is_streaming, return_trace_id
            )

        if is_streaming:
            return await self._handle_stream_request(request_data, return_trace_id)
        return await self._handle_invoke_request(request_data, return_trace_id)

    async def _handle_background_request(
        self,
        request_data: dict[str, Any],
        is_streaming: bool,
        return_trace_id: bool,
    ) -> dict[str, Any] | StreamingResponse:
        """Start a new conversation and return response_id immediately."""
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        await create_response(response_id, "in_progress")

        logger.debug(
            "Background response created",
            extra={"response_id": response_id, "stream": is_streaming},
        )

        response_obj: dict[str, Any] = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "error": None,
            "incomplete_details": None,
            "output": [],
            "metadata": {},
        }

        # Fire-and-forget is intentional — task status is persisted to the database.
        if is_streaming:
            asyncio.create_task(
                self._run_background_stream(response_id, request_data, return_trace_id)
            )
            return await self._handle_retrieve_request(
                response_id,
                stream=True,
                starting_after=0,
            )
        else:
            asyncio.create_task(
                self._run_background_invoke(response_id, request_data, return_trace_id)
            )
            return response_obj

    @asynccontextmanager
    async def _task_scope(
        self, response_id: str, state: dict[str, Any]
    ) -> AsyncGenerator[None, None]:
        """Timeout + error handling wrapper for background tasks."""
        try:
            async with asyncio.timeout(self._settings.task_timeout_seconds):
                yield
        except TimeoutError:
            logger.warning(
                "Task %s timed out after %ss",
                response_id,
                self._settings.task_timeout_seconds,
            )
            asyncio.create_task(
                _deferred_mark_failed(
                    response_id, delay=self._settings.cleanup_timeout_seconds
                ),
                name=f"deferred-fail-{response_id}",
            )
        except Exception as exc:
            logger.exception("Task %s failed: %s", response_id, exc)
            try:
                # TODO: sequence number computation is racy (see _deferred_mark_failed).
                async with asyncio.timeout(self._settings.cleanup_timeout_seconds):
                    existing = await get_messages(response_id, after_sequence=None)
                    next_seq = max((seq for seq, _, _ in existing), default=-1) + 1
                    await append_message(
                        response_id,
                        next_seq,
                        item=None,
                        stream_event={
                            "type": "error",
                            "error": {
                                "message": str(exc),
                                "type": "server_error",
                                "code": "task_failed",
                            },
                        },
                    )
                    await update_response_status(response_id, "failed")
            except Exception:
                logger.exception(
                    "[error-cleanup] Immediate update failed for %s, deferring",
                    response_id,
                )
                asyncio.create_task(
                    _deferred_mark_failed(
                        response_id,
                        delay=self._settings.cleanup_timeout_seconds,
                        reason=str(exc),
                    ),
                    name=f"deferred-fail-{response_id}",
                )

    async def _run_background_stream(
        self,
        response_id: str,
        request_data: dict[str, Any],
        return_trace_id: bool = False,
    ) -> None:
        """Timeout-guarded wrapper around the streaming agent loop."""
        state: dict[str, Any] = {"seq": 0}
        async with self._task_scope(response_id, state):
            await self._do_background_stream(response_id, request_data, return_trace_id, state)

    def transform_stream_event(self, event: dict, response_id: str) -> dict:
        """Override to transform events before persistence (e.g. replace placeholder IDs)."""
        return event

    async def _do_background_stream(
        self,
        response_id: str,
        request_data: dict[str, Any],
        return_trace_id: bool,
        state: dict[str, Any],
    ) -> None:
        """Run agent via stream_fn, persist each stream event as a message row."""
        stream_fn = get_stream_function()
        if stream_fn is None:
            await update_response_status(response_id, "failed")
            raise RuntimeError("No stream function registered; cannot run background stream")

        func_name = stream_fn.__name__
        all_chunks: list[dict[str, Any]] = []
        seq = 0

        with mlflow.start_span(name=f"{func_name}") as span:
            span.set_inputs(request_data)
            async for event in stream_fn(request_data):
                evt = self.validator.validate_and_convert_result(event, stream=True)
                evt = self.transform_stream_event(evt, response_id)

                all_chunks.append(evt)
                item = evt.get("item")
                evt_type = evt.get("type", "message")
                logger.debug(
                    "SSE event (background)",
                    extra={"response_id": response_id, "seq": seq, "type": evt_type},
                )
                await append_message(
                    response_id,
                    seq,
                    item=json.dumps(item) if item is not None else None,
                    stream_event=evt,
                )
                seq += 1
                state["seq"] = seq

            span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "openai")
            span.set_outputs(ResponsesAgent.responses_agent_output_reducer(all_chunks))

            if return_trace_id:
                await append_message(
                    response_id,
                    seq,
                    stream_event={"trace_id": span.trace_id},
                )

        await update_response_status(response_id, "completed")
        logger.debug(
            "Background stream completed",
            extra={"response_id": response_id, "total_events": seq},
        )

    async def _run_background_invoke(
        self,
        response_id: str,
        request_data: dict[str, Any],
        return_trace_id: bool = False,
    ) -> None:
        """Timeout-guarded wrapper around the invoke agent loop."""
        state: dict[str, Any] = {"seq": 0}
        async with self._task_scope(response_id, state):
            await self._do_background_invoke(response_id, request_data, return_trace_id, state)

    async def _do_background_invoke(
        self,
        response_id: str,
        request_data: dict[str, Any],
        return_trace_id: bool,
        state: dict[str, Any],
    ) -> None:
        """Run agent via invoke_fn, persist each output item as a message row."""
        invoke_fn = get_invoke_function()
        if invoke_fn is None:
            await update_response_status(response_id, "failed")
            raise RuntimeError("No invoke function registered; cannot run background invoke")

        func_name = invoke_fn.__name__

        with mlflow.start_span(name=f"{func_name}") as span:
            span.set_inputs(request_data)
            if inspect.iscoroutinefunction(invoke_fn):
                result = await invoke_fn(request_data)
            else:
                result = invoke_fn(request_data)

            result = self.validator.validate_and_convert_result(result)
            span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "openai")
            span.set_outputs(result)

        output = result.get("output", [])
        for i, item in enumerate(output):
            item_dict = (
                item
                if isinstance(item, dict)
                else (item.model_dump() if hasattr(item, "model_dump") else {"content": str(item)})
            )
            await append_message(
                response_id,
                i,
                item=json.dumps(item_dict),
                stream_event={"type": "response.output_item.done", "item": item_dict},
            )
            state["seq"] = i + 1
        if return_trace_id:
            await update_response_trace_id(response_id, span.trace_id)
        await update_response_status(response_id, "completed")
        logger.debug(
            "Background invoke completed",
            extra={"response_id": response_id, "output_items": len(output)},
        )

    async def _handle_retrieve_request(
        self,
        response_id: str,
        stream: bool,
        starting_after: int,
    ) -> dict[str, Any] | StreamingResponse:
        """Poll or stream messages from the database for a given response_id.

        Args:
            starting_after: Sequence number to resume from. 0 means fetch all
                messages (sequence numbers start at 0). Values > 0 fetch only
                messages with sequence_number > starting_after.
        """
        resp = await get_response(response_id)
        if resp is None:
            raise HTTPException(status_code=404, detail="Response not found")

        _, status, created_at, trace_id = resp

        if (
            status == "in_progress"
            and _age_seconds(created_at) > self._settings.task_timeout_seconds
        ):
            # Use conditional update so only one concurrent request performs the transition.
            updated = await update_response_status(
                response_id, "failed", expected_current_status="in_progress"
            )
            if updated:
                logger.warning(
                    "Stale in_progress run detected, marking as failed",
                    extra={
                        "response_id": response_id,
                        "age_s": _age_seconds(created_at),
                    },
                )
                # TODO: sequence number computation here is racy under concurrent writers.
                # Acceptable at current scale; for high-QPS use a DB-assigned sequence or
                # SELECT FOR UPDATE on the response row to serialise writers.
                existing = await get_messages(response_id, after_sequence=None)
                next_seq = max((seq for seq, _, _ in existing), default=-1) + 1
                await append_message(
                    response_id,
                    next_seq,
                    item=None,
                    stream_event={
                        "type": "error",
                        "error": {
                            "message": "Task timed out",
                            "type": "server_error",
                            "code": "task_timeout",
                        },
                    },
                )
            status = "failed"

        logger.debug(
            "Retrieve request",
            extra={
                "response_id": response_id,
                "stream": stream,
                "starting_after": starting_after,
                "status": status,
            },
        )

        if stream:
            return StreamingResponse(
                self._stream_retrieve(response_id, starting_after),
                media_type="text/event-stream",
            )

        messages = await get_messages(response_id, after_sequence=None)
        if not messages and status == "in_progress":
            return {"id": response_id, "status": "in_progress"}
        if status == "completed" and messages:
            output = []
            for _, _, evt in messages:
                if evt and "item" in evt:
                    output.append(evt["item"])
            result: dict[str, Any] = {
                "id": response_id,
                "status": "completed",
                "output": output,
            }
            if trace_id:
                result["metadata"] = {"trace_id": trace_id}
            return result
        if status == "failed" and messages:
            for _, _, evt in messages:
                if evt and evt.get("type") == "error":
                    return {"id": response_id, "status": "failed", "error": evt.get("error")}
        return {"id": response_id, "status": status}

    async def _stream_retrieve(
        self,
        response_id: str,
        starting_after: int,
    ) -> AsyncGenerator[str, None]:
        """Stream messages as SSE events from the database.

        Args:
            starting_after: Sequence number to resume from. 0 means fetch all
                messages. Values > 0 fetch only messages after that sequence.
        """
        poll_interval = self._settings.poll_interval_seconds
        last_seq = starting_after
        deadline = time.monotonic() + self._settings.task_timeout_seconds

        while time.monotonic() < deadline:
            logger.debug(
                "Poll iteration for %s (last_seq=%s)",
                response_id,
                last_seq,
            )
            resp = await get_response(response_id)
            if resp is None:
                logger.debug(
                    "SSE error event",
                    extra={"response_id": response_id, "error": "response_not_found"},
                )
                yield _sse_event(
                    "error",
                    {
                        "error": {
                            "message": "Response not found",
                            "type": "not_found",
                            "code": "response_not_found",
                        }
                    },
                )
                break

            _, status, _, _ = resp
            # starting_after=0 fetches all messages (sequence numbers start at 0).
            # We use after_sequence=-1 for the DB query so that seq 0 is included.
            after_seq = last_seq - 1 if last_seq == 0 else last_seq
            messages = await get_messages(response_id, after_sequence=after_seq)

            for seq, _, evt in messages:
                if evt is not None:
                    evt = {**evt, "sequence_number": seq}
                    event_type = evt.get("type", "message")
                    logger.debug(
                        "SSE event",
                        extra={"response_id": response_id, "seq": seq, "type": event_type},
                    )
                    yield _sse_event(event_type, evt)
                last_seq = seq

            if status == "completed":
                logger.debug(
                    "SSE stream ended",
                    extra={"response_id": response_id, "status": "completed"},
                )
                yield "data: [DONE]\n\n"
                break

            if status == "failed":
                logger.debug(
                    "SSE stream ended",
                    extra={"response_id": response_id, "status": "failed"},
                )
                break

            await asyncio.sleep(poll_interval)
        else:
            # Loop exited because we hit the deadline.
            logger.warning(
                "Stream retrieve timed out for %s after %ss",
                response_id,
                self._settings.task_timeout_seconds,
            )
            yield _sse_event(
                "error",
                {
                    "error": {
                        "message": "Stream retrieve timed out",
                        "type": "server_error",
                        "code": "stream_timeout",
                    }
                },
            )
