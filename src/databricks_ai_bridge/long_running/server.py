"""Long-running agent server with Lakebase persistence and background mode."""

import sys

if sys.version_info < (3, 11):
    raise RuntimeError("The long_running module requires Python 3.11 or later.")

import asyncio
import copy
import inspect
import json
import logging
import os
import socket
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
    claim_stale_response,
    create_response,
    get_messages,
    get_response,
    heartbeat_response,
    update_response_status,
    update_response_trace_id,
)
from databricks_ai_bridge.long_running.settings import LongRunningSettings
from databricks_ai_bridge.tool_repair import sanitize_tool_items
from databricks_ai_bridge.utils.annotations import experimental

logger = logging.getLogger(__name__)

BACKGROUND_KEY = "background"

# One ID per process so heartbeats + claims have a stable owner identity.
_POD_ID = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


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
            next_seq = max((seq for seq, _, _, _ in existing), default=-1) + 1
            attempt = await _current_attempt(response_id)

            error_event = {
                "type": "error",
                "error": {
                    "message": reason,
                    "type": "server_error",
                    "code": "task_timeout",
                },
            }
            await append_message(
                response_id,
                next_seq,
                item=None,
                stream_event=error_event,
                attempt_number=attempt,
            )
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


async def _current_attempt(response_id: str) -> int:
    """Fetch the current attempt_number for a response, defaulting to 1."""
    resp = await get_response(response_id)
    return resp.attempt_number if resp else 1


def _sse_event(event_type: str, data: dict[str, Any] | str) -> str:
    """Emit ``data:``-only SSE frames. Match the non-durable stream format
    so downstream SSE parsers dispatch on the payload's ``type`` field
    rather than a leading ``event:`` name line. Claude's multi-response
    stream (one response.created/completed pair per tool iteration) plus
    the event-name prefix confuses the AI SDK's Databricks provider into
    a retry loop."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"data: {payload}\n\n"


def _age_seconds(created_at: datetime) -> float:
    """Return the age of a timestamp in seconds."""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (now - created_at).total_seconds()


# Tool-pair items are collected into the main inheritance bucket and
# preserved in event-log order. Narrative ``message`` items are routed
# separately in the collector (see ``_collect_prior_attempt_tool_events``)
# so they can be hoisted past the tool pairs for Anthropic adapter
# compatibility — adding ``"message"`` here would break that hoist.
_TOOL_PAIR_TYPES = ("function_call", "function_call_output")


def _iter_attempt_events(messages: list[tuple], attempt: int):
    """Yield ``(event_type, event_dict)`` pairs for the given attempt.

    Skips rows from other attempts and non-dict event payloads so callers
    can write single-concern walkers without repeating the same filter.
    """
    for _seq, _item_json, evt, attempt_tag in messages:
        if attempt_tag != attempt:
            continue
        if not isinstance(evt, dict):
            continue
        yield evt.get("type"), evt


def _extract_completed_items(messages: list[tuple], attempt: int) -> tuple[list[dict], list[dict]]:
    """Scan ``.done`` events and partition into (tool pairs, narrative)."""
    tool_items: list[dict] = []
    narrative_items: list[dict] = []
    for t, evt in _iter_attempt_events(messages, attempt):
        if t != "response.output_item.done":
            continue
        item = evt.get("item")
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            narrative_items.append(item)
        elif itype in _TOOL_PAIR_TYPES:
            tool_items.append(item)
    return tool_items, narrative_items


def _reassemble_partial_message(messages: list[tuple], attempt: int) -> dict | None:
    """Return a synthetic assistant message if the attempt ended with a
    never-completed in-flight text item, else ``None``.

    Tracks ``output_item.added`` for message items, accumulates their
    ``output_text.delta`` frames, and drops the tracker when a matching
    ``.done`` arrives (that item is authoritative). Anything left at the
    end is an unfinished message whose deltas we stitch into a synthetic
    item so the next attempt's LLM can continue the prior narration.
    """
    in_progress_text: dict[str, list[str]] = {}
    in_progress_order: list[str] = []
    for t, evt in _iter_attempt_events(messages, attempt):
        if t == "response.output_item.added":
            item = evt.get("item")
            if isinstance(item, dict) and item.get("type") == "message":
                iid = item.get("id")
                if iid:
                    in_progress_text.setdefault(iid, [])
                    if iid not in in_progress_order:
                        in_progress_order.append(iid)
        elif t == "response.output_item.done":
            item = evt.get("item")
            if isinstance(item, dict):
                iid = item.get("id")
                if iid in in_progress_text:
                    del in_progress_text[iid]
                    in_progress_order.remove(iid)
        elif t == "response.output_text.delta":
            iid = evt.get("item_id")
            delta = evt.get("delta")
            if iid and isinstance(delta, str) and iid in in_progress_text:
                in_progress_text[iid].append(delta)

    for iid in in_progress_order:
        chunks = in_progress_text.get(iid) or []
        if not chunks:
            continue
        return {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "".join(chunks), "annotations": []},
            ],
        }
    return None


def _collect_prior_attempt_tool_events(
    messages: list[tuple], prior_attempt_number: int
) -> list[dict]:
    """Return items the given prior attempt emitted, reordered to be a
    valid provider message sequence on replay.

    Composition: (completed tool pairs in event order) → (completed
    narrative messages in event order) → (partial reassembled message, if
    any). Claude's raw stream interleaves narrative ``message`` items
    between each ``function_call`` and its ``function_call_output``, which
    in Anthropic format would look like ``assistant(tool_use)`` →
    ``assistant(text)`` → ``user(tool_result)`` and trip the provider's
    "tool_use must be immediately followed by tool_result" rule (HTTP
    400). Hoisting narrative past the tool pairs keeps each function_call
    adjacent to its output and lets the narrative flow as a trailing
    assistant block.

    ``messages`` is the repository's tuples ``(seq, item_json,
    stream_event, attempt_number)``.
    """
    tool_items, narrative_items = _extract_completed_items(messages, prior_attempt_number)
    partial = _reassemble_partial_message(messages, prior_attempt_number)
    if partial is not None:
        narrative_items.append(partial)
    return tool_items + narrative_items


def _sanitize_request_input(request_dict: dict[str, Any]) -> dict[str, Any]:
    """Reconcile orphaned function_call / function_call_output items in
    ``request['input']`` via the shared :func:`sanitize_tool_items` walker.

    Walks the whole history (not just the trailing turn) because UI-echoed
    history can carry orphans from prior crashed turns mid-list.
    Mutates ``request_dict['input']`` in place and returns the dict for
    caller convenience.
    """
    items = request_dict.get("input")
    if not isinstance(items, list) or not items:
        return request_dict
    request_dict["input"] = sanitize_tool_items(items, log_prefix="[durable] input sanitized")
    return request_dict


def _rotate_conversation_id(
    request_dict: dict[str, Any],
    new_attempt_number: int,
    response_id: str,
) -> dict[str, Any]:
    """Rotate the conversation anchor to a per-attempt value.

    After a crash, attempt N+1 should see a FRESH checkpointer / session so it
    doesn't inherit mid-turn state that the SDK can't repair cleanly (most
    notably the LangGraph stream-event attempt-boundary orphan artifact).
    The handler's priority chain is:

        1. custom_inputs.thread_id / session_id   (explicit, wins)
        2. context.conversation_id                (fallback)
        3. auto-generated                         (last resort)

    We drop (1), pick the current base anchor, and write ``{base}::attempt-N``
    into (2). The handler then resolves to a fresh key for this attempt while
    still being deterministic across retries of the same attempt.

    The LLM sees full turn history via ``original_request.input``, which was
    captured at the initial POST — before any attempt ran, so it's clean by
    construction.
    """
    custom_inputs = request_dict.get("custom_inputs")
    if not isinstance(custom_inputs, dict):
        custom_inputs = {}

    base_anchor = (
        custom_inputs.get("thread_id")
        or custom_inputs.get("session_id")
        or (request_dict.get("context") or {}).get("conversation_id")
        or response_id
    )

    custom_inputs.pop("thread_id", None)
    custom_inputs.pop("session_id", None)
    request_dict["custom_inputs"] = custom_inputs

    ctx = request_dict.get("context") or {}
    ctx = dict(ctx)
    rotated = f"{base_anchor}::attempt-{new_attempt_number}"
    ctx["conversation_id"] = rotated
    request_dict["context"] = ctx
    logger.info(
        "[durable] rotated conversation_id for resume response_id=%s attempt=%d base=%s rotated=%s",
        response_id,
        new_attempt_number,
        base_anchor,
        rotated,
    )
    return request_dict


def _inject_conversation_id(request_dict: dict[str, Any], response_id: str) -> dict[str, Any]:
    """Anchor the request to ``response_id`` as its conversation.

    Operates on a plain dict — the caller is responsible for converting to/from
    pydantic via ``model_dump()`` and the server's validator.

    Templates that back this server use ``context.conversation_id`` (and
    ``custom_inputs.thread_id`` / ``custom_inputs.session_id``) as priority-2
    fallbacks to derive their stateful thread/session key. If neither is
    provided by the client, a resumed invocation from another pod would
    generate a *fresh* ID and miss the checkpoint entirely — so we stamp the
    conversation_id here before persisting the request, guaranteeing that
    every replay hits the same memory store.

    Client-supplied values take precedence and are left untouched.
    """
    out = copy.deepcopy(request_dict) if request_dict else {}
    custom_inputs = out.get("custom_inputs") or {}
    if custom_inputs.get("thread_id") or custom_inputs.get("session_id"):
        return out
    ctx = out.get("context") or {}
    if ctx.get("conversation_id"):
        return out
    ctx = dict(ctx)
    ctx["conversation_id"] = response_id
    out["context"] = ctx
    return out


@experimental
class LongRunningAgentServer(AgentServer):
    """AgentServer subclass adding background mode, retrieve endpoints, and
    durable resume.

    Only compatible with ``ResponsesAgent`` mode.

    Background mode requires a Lakebase database for persistence. The database
    connection can be configured via constructor arguments or environment variables:

        - ``LAKEBASE_INSTANCE_NAME``: Provisioned Lakebase instance name.
        - ``LAKEBASE_AUTOSCALING_ENDPOINT``: Lakebase autoscaling endpoint URL.
        - ``LAKEBASE_AUTOSCALING_PROJECT``: Lakebase autoscaling project (must be
          set together with ``LAKEBASE_AUTOSCALING_BRANCH``).
        - ``LAKEBASE_AUTOSCALING_BRANCH``: Lakebase autoscaling branch (must be
          set together with ``LAKEBASE_AUTOSCALING_PROJECT``).

    At least one of the following must be set to enable background mode:
    ``LAKEBASE_INSTANCE_NAME``, ``LAKEBASE_AUTOSCALING_ENDPOINT``, or both
    ``LAKEBASE_AUTOSCALING_PROJECT`` and ``LAKEBASE_AUTOSCALING_BRANCH``.

    Durable resume: when ``GET /responses/{id}`` sees an ``in_progress`` run
    whose owning pod has stopped heartbeating for more than
    ``heartbeat_stale_threshold_seconds``, the retrieving pod atomically claims
    the run and re-invokes the registered handler with a rotated
    ``conversation_id`` (so the agent SDK resolves to a fresh thread/session),
    the original request's ``input`` enriched with the prior attempt's already
    emitted tool calls / outputs / narrative, and an ``[INTERRUPTED]`` synthetic
    output paired with any tool call that didn't finish. Completed work is
    preserved; only the interrupted step re-runs.

    Args:
        enable_chat_proxy: Whether to enable the chat proxy endpoint.
        db_instance_name: Lakebase provisioned instance name. Overrides
            ``LAKEBASE_INSTANCE_NAME``.
        db_autoscaling_endpoint: Lakebase autoscaling endpoint URL. Overrides
            ``LAKEBASE_AUTOSCALING_ENDPOINT``.
        db_project: Lakebase autoscaling project. Overrides
            ``LAKEBASE_AUTOSCALING_PROJECT``.
        db_branch: Lakebase autoscaling branch. Overrides
            ``LAKEBASE_AUTOSCALING_BRANCH``.
        task_timeout_seconds: Max time for a background task before timeout.
            Defaults to 3600 (1 hour).
        poll_interval_seconds: Interval between DB polls when streaming.
            Defaults to 1.0.
        db_statement_timeout_ms: Postgres statement timeout.
            Defaults to 5000 (5 seconds).
        cleanup_timeout_seconds: Timeout for DB cleanup after task failure.
            Defaults to 7.0.
        heartbeat_interval_seconds: How often the owning pod writes
            ``heartbeat_at`` while a run is in flight. Defaults to 3.0.
        heartbeat_stale_threshold_seconds: Age at which a heartbeat is
            considered stale and another pod may claim the run. Also used
            as the grace window for a freshly-created run that hasn't
            written its first heartbeat yet. Defaults to 10.0.
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
        heartbeat_interval_seconds: float = 3.0,
        heartbeat_stale_threshold_seconds: float = 10.0,
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
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            heartbeat_stale_threshold_seconds=heartbeat_stale_threshold_seconds,
        )
        self._db_instance_name = db_instance_name
        self._db_autoscaling_endpoint = db_autoscaling_endpoint
        self._db_project = db_project
        self._db_branch = db_branch
        # Track in-flight background tasks per response_id so the debug-kill
        # endpoint can simulate a pod crash without tearing the whole pod
        # down. Not load-bearing for correctness — durability still relies on
        # DB state, this is just a test affordance.
        self._running_tasks: dict[str, asyncio.Task] = {}
        super().__init__(agent_type, enable_chat_proxy=enable_chat_proxy)

    def _setup_routes(self) -> None:
        """Register routes. Reuses parent's POST /invocations and POST /responses.

        Adds GET /responses/{id} for polling/streaming when DB is configured.
        Auto-registers startup/shutdown events for DB lifecycle.
        """
        super()._setup_routes()

        @self.app.post("/responses/{response_id}/cancel")
        async def cancel_endpoint(response_id: str):
            raise HTTPException(
                status_code=501,
                detail="Cancellation is not yet implemented.",
            )

        # Debug endpoint for testing durable resume: cancels the in-flight
        # asyncio task that owns the given response_id WITHOUT running the
        # _task_scope cleanup, so the DB row stays in_progress with a
        # going-stale heartbeat — exactly the shape a real pod crash leaves.
        # Opt-in via env var so it's never exposed in production.
        if os.getenv("LONG_RUNNING_ENABLE_DEBUG_KILL") == "1":

            @self.app.post("/_debug/kill_task/{response_id}")
            async def _debug_kill_task(response_id: str):
                task = self._running_tasks.get(response_id)
                if task is None:
                    logger.info(
                        "[durable] kill endpoint: no task response_id=%s on pod=%s",
                        response_id,
                        _POD_ID,
                    )
                    raise HTTPException(
                        status_code=404,
                        detail=(
                            "No in-flight task for that response_id on this pod "
                            "(may already have finished or be running on another pod)."
                        ),
                    )
                logger.info(
                    "[durable] kill endpoint: cancelling task response_id=%s pod=%s",
                    response_id,
                    _POD_ID,
                )
                task.cancel()
                return {
                    "response_id": response_id,
                    "pod_id": _POD_ID,
                    "status": "task_cancelled",
                }

        db_configured = is_db_configured()

        @self.app.get("/responses/{response_id}")
        async def retrieve_endpoint(
            response_id: str,
            stream: bool = Query(False, description="Stream results as SSE"),
            starting_after: int = Query(
                0, ge=0, description="Resume from sequence number (0 means fetch all)"
            ),
        ):
            if not db_configured:
                raise HTTPException(
                    status_code=501,
                    detail="Response retrieval requires a database configuration.",
                )
            if starting_after != 0 and not stream:
                raise HTTPException(
                    status_code=400,
                    detail="starting_after is only supported when stream=true.",
                )
            return await self._handle_retrieve_request(
                response_id,
                stream=stream,
                starting_after=starting_after,
            )

        if not db_configured:
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
        return_trace_id = (get_request_headers().get(RETURN_TRACE_HEADER) or "").lower() == "true"

        if self._settings.auto_sanitize_input:
            data = _sanitize_request_input(data)

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
        # Anchor the conversation to response_id so any future replay from a
        # different pod resolves to the same agent-SDK thread/session. We
        # round-trip through dict + validator so the handler still receives a
        # pydantic ResponsesAgentRequest (its declared arg type). The
        # declared param type is ``dict`` but the runtime object is a pydantic
        # model from ``validate_and_convert_request``; fall back to ``dict()``
        # when tests pass a plain dict directly.
        dump = getattr(request_data, "model_dump", None)
        request_dict = dump() if callable(dump) else dict(request_data)
        durable_dict = _inject_conversation_id(request_dict, response_id)
        durable_request = self.validator.validate_and_convert_request(durable_dict)
        await create_response(
            response_id,
            "in_progress",
            owner_pod_id=_POD_ID,
            original_request=durable_dict,
        )

        logger.info(
            "Background response created response_id=%s stream=%s pod=%s",
            response_id,
            is_streaming,
            _POD_ID,
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
        # We still track the task handle so the debug-kill endpoint can simulate
        # a crash (and so we know whether a claim target lives on this pod).
        if is_streaming:
            task = asyncio.create_task(
                self._run_background_stream(
                    response_id, durable_request, return_trace_id, attempt_number=1
                )
            )
            self._track_task(response_id, task)
            return await self._handle_retrieve_request(
                response_id,
                stream=True,
                starting_after=0,
            )
        else:
            task = asyncio.create_task(
                self._run_background_invoke(
                    response_id, durable_request, return_trace_id, attempt_number=1
                )
            )
            self._track_task(response_id, task)
            return response_obj

    def _track_task(self, response_id: str, task: asyncio.Task) -> None:
        """Record a background task so the debug-kill endpoint can find it."""
        self._running_tasks[response_id] = task
        task.add_done_callback(lambda _t: self._running_tasks.pop(response_id, None))

    @asynccontextmanager
    async def _heartbeat(self, response_id: str) -> AsyncGenerator[None, None]:
        """Keep the response row's heartbeat_at fresh while the body runs.

        A background task writes ``heartbeat_at = now()`` every
        ``heartbeat_interval_seconds`` for the owning pod. It stops when the
        body returns/raises. Heartbeat write failures are logged but do not
        interrupt the agent run — the stale-run check will detect a dead pod.
        """
        interval = self._settings.heartbeat_interval_seconds
        stop = asyncio.Event()

        async def _beat():
            beats = 0
            logger.info(
                "[durable] heartbeat start response_id=%s pod=%s interval=%.1fs",
                response_id,
                _POD_ID,
                interval,
            )
            try:
                while not stop.is_set():
                    try:
                        await heartbeat_response(response_id, _POD_ID)
                        beats += 1
                        # Sampled heartbeat log so the lifecycle is visible
                        # without spamming every interval. Every 5th (~15s
                        # at 3s interval) is a good compromise.
                        if beats % 5 == 1:
                            logger.info(
                                "[durable] heartbeat beat#%d response_id=%s pod=%s",
                                beats,
                                response_id,
                                _POD_ID,
                            )
                    except Exception:
                        logger.warning(
                            "[durable] heartbeat write failed response_id=%s; will retry",
                            response_id,
                            exc_info=True,
                        )
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=interval)
                    except TimeoutError:
                        pass
            except asyncio.CancelledError:
                pass
            logger.info(
                "[durable] heartbeat stop response_id=%s pod=%s total_beats=%d",
                response_id,
                _POD_ID,
                beats,
            )

        hb_task = asyncio.create_task(_beat(), name=f"heartbeat-{response_id}")
        try:
            yield
        finally:
            stop.set()
            hb_task.cancel()
            try:
                await hb_task
            except (asyncio.CancelledError, Exception):
                pass

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
                _deferred_mark_failed(response_id, delay=self._settings.cleanup_timeout_seconds),
                name=f"deferred-fail-{response_id}",
            )
        except Exception as exc:
            logger.exception("Task %s failed: %s", response_id, exc)
            try:
                # TODO: sequence number computation is racy (see _deferred_mark_failed).
                async with asyncio.timeout(self._settings.cleanup_timeout_seconds):
                    existing = await get_messages(response_id, after_sequence=None)
                    next_seq = max((seq for seq, _, _, _ in existing), default=-1) + 1
                    attempt = await _current_attempt(response_id)
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
                        attempt_number=attempt,
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
        *,
        attempt_number: int = 1,
    ) -> None:
        """Timeout-guarded wrapper around the streaming agent loop."""
        state: dict[str, Any] = {"seq": 0}
        async with self._task_scope(response_id, state), self._heartbeat(response_id):
            await self._do_background_stream(
                response_id,
                request_data,
                return_trace_id,
                state,
                attempt_number=attempt_number,
            )

    def transform_stream_event(self, event: dict, response_id: str) -> dict:
        """Override to transform events before persistence (e.g. replace placeholder IDs)."""
        return event

    async def _do_background_stream(
        self,
        response_id: str,
        request_data: dict[str, Any],
        return_trace_id: bool,
        state: dict[str, Any],
        *,
        attempt_number: int = 1,
    ) -> None:
        """Run agent via stream_fn, persist each stream event as a message row."""
        stream_fn = get_stream_function()
        if stream_fn is None:
            await update_response_status(response_id, "failed")
            raise RuntimeError("No stream function registered; cannot run background stream")

        func_name = stream_fn.__name__
        logger.info(
            "[durable] background stream start response_id=%s attempt=%d pod=%s handler=%s",
            response_id,
            attempt_number,
            _POD_ID,
            func_name,
        )
        all_chunks: list[dict[str, Any]] = []
        # Continue sequence numbering across attempts so the client's cursor
        # never rewinds on resume. First attempt starts at 0 and skips the DB
        # lookup — keeps the fast path identical to pre-resume behavior and
        # avoids an extra query per background request.
        if attempt_number > 1:
            existing = await get_messages(response_id, after_sequence=None)
            seq = max((s for s, _, _, _ in existing), default=-1) + 1
        else:
            seq = 0

        with mlflow.start_span(name=func_name) as span:
            span.set_inputs(request_data)
            async for event in stream_fn(request_data):
                evt = self.validator.validate_and_convert_result(event, stream=True)
                evt = self.transform_stream_event(evt, response_id)

                all_chunks.append(evt)
                item = evt.get("item")
                evt_type = evt.get("type", "message")
                logger.debug(
                    "SSE event (background)",
                    extra={
                        "response_id": response_id,
                        "seq": seq,
                        "type": evt_type,
                        "attempt": attempt_number,
                    },
                )
                await append_message(
                    response_id,
                    seq,
                    item=json.dumps(item) if item is not None else None,
                    stream_event=evt,
                    attempt_number=attempt_number,
                )
                seq += 1
                state["seq"] = seq
                # Explicit yield so task.cancel() propagates promptly on
                # tight event streams. The OpenAI Agents Runner's
                # stream_events() awaits a queue that empties fast enough
                # that cancellation can sit for tens of seconds without this.
                await asyncio.sleep(0)

            span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "openai")
            span.set_outputs(ResponsesAgent.responses_agent_output_reducer(all_chunks))

            if return_trace_id:
                await append_message(
                    response_id,
                    seq,
                    stream_event={"trace_id": span.trace_id},
                    attempt_number=attempt_number,
                )

        await update_response_status(response_id, "completed")
        logger.info(
            "[durable] background stream completed response_id=%s attempt=%d "
            "total_events=%d pod=%s",
            response_id,
            attempt_number,
            seq,
            _POD_ID,
        )

    async def _run_background_invoke(
        self,
        response_id: str,
        request_data: dict[str, Any],
        return_trace_id: bool = False,
        *,
        attempt_number: int = 1,
    ) -> None:
        """Timeout-guarded wrapper around the invoke agent loop."""
        state: dict[str, Any] = {"seq": 0}
        async with self._task_scope(response_id, state), self._heartbeat(response_id):
            await self._do_background_invoke(
                response_id,
                request_data,
                return_trace_id,
                state,
                attempt_number=attempt_number,
            )

    async def _do_background_invoke(
        self,
        response_id: str,
        request_data: dict[str, Any],
        return_trace_id: bool,
        state: dict[str, Any],
        *,
        attempt_number: int = 1,
    ) -> None:
        """Run agent via invoke_fn, persist each output item as a message row."""
        invoke_fn = get_invoke_function()
        if invoke_fn is None:
            await update_response_status(response_id, "failed")
            raise RuntimeError("No invoke function registered; cannot run background invoke")

        func_name = invoke_fn.__name__

        with mlflow.start_span(name=func_name) as span:
            span.set_inputs(request_data)
            if inspect.iscoroutinefunction(invoke_fn):
                result = await invoke_fn(request_data)
            else:
                result = invoke_fn(request_data)

            result = self.validator.validate_and_convert_result(result)
            span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "openai")
            span.set_outputs(result)

        output = result.get("output", [])
        # Continue sequence numbering across attempts (see _do_background_stream).
        if attempt_number > 1:
            existing = await get_messages(response_id, after_sequence=None)
            base_seq = max((s for s, _, _, _ in existing), default=-1) + 1
        else:
            base_seq = 0
        for i, item in enumerate(output):
            item_dict = (
                item
                if isinstance(item, dict)
                else (item.model_dump() if hasattr(item, "model_dump") else {"content": str(item)})
            )
            seq = base_seq + i
            await append_message(
                response_id,
                seq,
                item=json.dumps(item_dict),
                stream_event={"type": "response.output_item.done", "item": item_dict},
                attempt_number=attempt_number,
            )
            state["seq"] = seq + 1
        if return_trace_id:
            await update_response_trace_id(response_id, span.trace_id)
        await update_response_status(response_id, "completed")
        logger.debug(
            "Background invoke completed",
            extra={"response_id": response_id, "output_items": len(output)},
        )

    async def _try_claim_and_resume(self, response_id: str, resp) -> int | None:
        """If ``resp`` is a stale in-progress run, attempt an atomic claim.

        On success, kick off a new background task that re-invokes the handler
        on a rotated conversation anchor with the replayed input enriched by
        the prior attempt's emitted items, and returns the new
        ``attempt_number``. On failure (another pod won, or the run is no
        longer stale), returns ``None``.

        This is the lazy resume path: triggered by a client retrieve. Pods
        don't poll for stale work proactively in v1 — if no client ever calls
        ``GET /responses/{id}``, the task_timeout sweep eventually marks it
        failed.
        """
        if resp.status != "in_progress":
            return None
        # The run may be freshly started but too young to have a heartbeat yet;
        # respect the creation age as a grace period equal to the stale
        # threshold. Otherwise a quick follow-up retrieve could hijack a
        # running pod before it ever writes its first heartbeat.
        if resp.heartbeat_at is None:
            age = _age_seconds(resp.created_at)
            if age < self._settings.heartbeat_stale_threshold_seconds:
                logger.debug(
                    "[durable] claim skipped response_id=%s reason=grace_period "
                    "age=%.1fs threshold=%.1fs",
                    response_id,
                    age,
                    self._settings.heartbeat_stale_threshold_seconds,
                )
                return None
        else:
            hb_age = _age_seconds(resp.heartbeat_at)
            if hb_age < self._settings.heartbeat_stale_threshold_seconds:
                # Heartbeat is fresh — owner is alive. Common case, keep
                # quiet at debug so we don't spam every poll iteration.
                logger.debug(
                    "[durable] claim skipped response_id=%s reason=heartbeat_fresh "
                    "age=%.1fs threshold=%.1fs",
                    response_id,
                    hb_age,
                    self._settings.heartbeat_stale_threshold_seconds,
                )
                return None
            logger.info(
                "[durable] stale heartbeat detected response_id=%s "
                "heartbeat_age=%.1fs threshold=%.1fs current_owner=%s",
                response_id,
                hb_age,
                self._settings.heartbeat_stale_threshold_seconds,
                resp.owner_pod_id,
            )
        if resp.original_request is None:
            # Nothing to replay from — the run predates durability metadata.
            logger.warning(
                "[durable] cannot resume response_id=%s reason=no_original_request",
                response_id,
            )
            return None

        logger.info(
            "[durable] attempting claim response_id=%s current_attempt=%d new_owner=%s",
            response_id,
            resp.attempt_number,
            _POD_ID,
        )
        new_attempt = await claim_stale_response(
            response_id,
            new_owner_pod_id=_POD_ID,
            stale_threshold_seconds=self._settings.heartbeat_stale_threshold_seconds,
        )
        if new_attempt is None:
            # Someone else owns it, or the row was updated between the read and
            # the claim. Expected under contention.
            logger.info(
                "[durable] claim lost response_id=%s (another pod won or row changed)",
                response_id,
            )
            return None

        # Build a "resume" request by REPLAYING the original POST's input on a
        # ROTATED conversation anchor, enriched with the prior attempt's
        # already-emitted tool events:
        #
        # 1. Carry forward the prior attempt's function_call / function_call_output
        #    items so the LLM sees what's already been done. Without this,
        #    attempt N+1 re-plans from just the user's latest message and
        #    re-emits tool calls that previously completed (e.g. it re-runs
        #    get_time even though only deep_research was interrupted). The
        #    interrupted tool's orphan function_call gets a synthetic
        #    "interrupted" output via the sanitizer below.
        #
        # 2. Rotate conversation_id so the handler's SDK helpers resolve to a
        #    FRESH thread_id / session_id for this attempt. Without this, the
        #    handler would reload the crashed attempt's mid-turn checkpoint,
        #    which on LangGraph produces a stream-event orphan artifact at the
        #    attempt boundary (rotation-findings.md stress test). Attempt N+1
        #    runs on a clean checkpointer; the prior-attempt tool events in
        #    input[] are the single source of truth for what already ran.
        existing = await get_messages(response_id, after_sequence=None)
        next_seq = max((s for s, _, _, _ in existing), default=-1) + 1
        prior_tool_events = _collect_prior_attempt_tool_events(
            existing, prior_attempt_number=new_attempt - 1
        )

        resume_dict = copy.deepcopy(resp.original_request)
        if prior_tool_events:
            resume_input = list(resume_dict.get("input") or [])
            resume_input.extend(prior_tool_events)
            resume_dict["input"] = resume_input
            logger.info(
                "[durable] resume inherited %d tool-event item(s) from attempt %d response_id=%s",
                len(prior_tool_events),
                new_attempt - 1,
                response_id,
            )
        if self._settings.auto_sanitize_input:
            resume_dict = _sanitize_request_input(resume_dict)
        resume_dict = _rotate_conversation_id(resume_dict, new_attempt, response_id)
        resume_request = self.validator.validate_and_convert_request(resume_dict)
        await append_message(
            response_id,
            next_seq,
            stream_event={
                "type": "response.resumed",
                "attempt": new_attempt,
                "from_seq": next_seq,
            },
            attempt_number=new_attempt,
        )

        logger.info(
            "[durable] claim succeeded response_id=%s new_attempt=%d pod=%s resume_from_seq=%d",
            response_id,
            new_attempt,
            _POD_ID,
            next_seq,
        )

        task = asyncio.create_task(
            self._run_background_stream(
                response_id,
                resume_request,
                return_trace_id=False,
                attempt_number=new_attempt,
            ),
            name=f"resume-{response_id}-{new_attempt}",
        )
        self._track_task(response_id, task)
        return new_attempt

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

        # Try a lazy resume before falling back to the absolute-timeout sweep.
        # This gives us crash-recovery semantics: an idle client reconnecting
        # after a pod died will reclaim the run and resume it here instead of
        # just marking it failed.
        await self._try_claim_and_resume(response_id, resp)

        # Refresh after the potential resume: status / attempt_number may have changed.
        resp = await get_response(response_id)
        if resp is None:
            raise HTTPException(status_code=404, detail="Response not found")

        status = resp.status
        created_at = resp.created_at
        trace_id = resp.trace_id

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
                existing = await get_messages(response_id, after_sequence=None)
                next_seq = max((seq for seq, _, _, _ in existing), default=-1) + 1
                attempt = await _current_attempt(response_id)
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
                    attempt_number=attempt,
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
            return {
                "id": response_id,
                "status": "in_progress",
                "attempt_number": resp.attempt_number,
            }
        if status == "completed" and messages:
            # Only consider items from the final (successful) attempt so that
            # abandoned in-progress items from crashed attempts don't leak
            # into the authoritative response body. Completed output_item.done
            # events across attempts together make up the conversation — the
            # agent SDK's checkpointer guarantees done-items are not re-emitted
            # by later attempts, so this is a union with no duplicates.
            output = []
            for _, _, evt, _attempt in messages:
                if evt and evt.get("type") == "response.output_item.done":
                    output.append(evt.get("item"))
            result: dict[str, Any] = {
                "id": response_id,
                "status": "completed",
                "output": [o for o in output if o is not None],
                "attempt_number": resp.attempt_number,
            }
            if trace_id:
                result["metadata"] = {"trace_id": trace_id}
            return result
        if status == "failed" and messages:
            for _, _, evt, _attempt in messages:
                if evt and evt.get("type") == "error":
                    return {
                        "id": response_id,
                        "status": "failed",
                        "error": evt.get("error"),
                        "attempt_number": resp.attempt_number,
                    }
        return {
            "id": response_id,
            "status": status,
            "attempt_number": resp.attempt_number,
        }

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

            status = resp.status
            # Self-heal: if this response is still in_progress but its owning
            # pod has gone silent past heartbeat_stale_threshold, try to claim
            # + resume on this pod. A no-op if heartbeat is fresh or another
            # pod already won. Without this, a stream opened before the crash
            # would idle forever polling a dead run — since _try_claim_and_resume
            # is only triggered by the outer retrieve handler on fresh GETs.
            if status == "in_progress":
                await self._try_claim_and_resume(response_id, resp)

            # starting_after=0 fetches all messages (sequence numbers start at 0).
            # We use after_sequence=-1 for the DB query so that seq 0 is included.
            after_seq = last_seq - 1 if last_seq == 0 else last_seq
            messages = await get_messages(response_id, after_sequence=after_seq)

            for seq, _, evt, _attempt in messages:
                if evt is not None:
                    # Tag every SSE frame with the response_id so proxies /
                    # clients can discover it without parsing nested fields.
                    evt = {**evt, "sequence_number": seq, "response_id": response_id}
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
