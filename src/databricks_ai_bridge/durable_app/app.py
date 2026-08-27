"""ASGI application exposing one generic durable agent entrypoint."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from databricks_ai_bridge.durable_runtime import (
    DatabricksDurableRuntime,
    DurabilityStore,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
    InMemoryDurabilityStore,
    JsonObject,
    LakebaseDurabilityStore,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

RUN_ID_HEADER = "Idempotency-Key"
SESSION_ID_HEADER = "X-Routing-Key"

_BACKGROUND_KEY = "background"
_STREAM_KEY = "stream"
_SESSION_STORE_ENV = "AGENT_SESSION_STORE"
_SHARED_SESSION_STORE_PROJECT = "databricks-internal-lakebase-agent-session-store"
_DEFAULT_SESSION_STORE_BRANCH = "production"

DurableAgentEntrypoint = Callable[[JsonObject, "DurableAgentContext"], Awaitable[JsonObject]]


@dataclass(frozen=True)
class DurableAgentContext:
    """Stable run/session identifiers and attempt metadata for an agent."""

    run_id: str
    session_id: str
    attempt: int
    _execution_context: DurableExecutionContext

    @property
    def is_recovery(self) -> bool:
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        """Persist an event before it is streamed to the client."""
        return await self._execution_context.emit(event)


class DatabricksDurableApp:
    """Host one JSON entrypoint with background execution and replay.

    The request body is passed to the entrypoint unchanged. Runtime metadata lives in
    headers: ``Idempotency-Key`` identifies the durable run and ``X-Routing-Key`` identifies
    the agent session. Both are generated when omitted and returned as response headers.

    Local development uses an in-memory store automatically. A configured Lakebase endpoint,
    project/branch, or managed ``AGENT_SESSION_STORE`` switches the same app to durable storage.
    """

    def __init__(
        self,
        entrypoint: DurableAgentEntrypoint | None = None,
        *,
        durability_store: DurabilityStore | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = "databricks_durable_app",
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 0.1,
    ) -> None:
        self._entrypoint = entrypoint
        store = durability_store or self._default_store(
            autoscaling_endpoint=autoscaling_endpoint,
            project=project,
            branch=branch,
            workspace_client=workspace_client,
            schema=schema,
        )
        self.runtime = DatabricksDurableRuntime(
            self._execute,
            durability_store=store,
            heartbeat_seconds=heartbeat_seconds,
            stale_seconds=stale_seconds,
            scan_seconds=scan_seconds,
            poll_seconds=poll_seconds,
        )

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            await self.runtime.start()
            try:
                yield
            finally:
                await self.runtime.stop()

        self.asgi_app = FastAPI(lifespan=lifespan)
        self.asgi_app.add_api_route("/invocations", self._invoke, methods=["POST"])
        self.asgi_app.add_api_route("/invocations/{run_id}", self._get, methods=["GET"])
        self.asgi_app.add_api_route(
            "/invocations/{run_id}/events",
            self._events,
            methods=["GET"],
        )
        self.asgi_app.add_api_route("/health", self._health, methods=["GET"])
        self.asgi_app.add_api_route("/api/healthz", self._health, methods=["GET"])

    def entrypoint(self, function: DurableAgentEntrypoint) -> DurableAgentEntrypoint:
        """Register the single agent function invoked for every durable attempt."""
        if self._entrypoint is not None:
            raise RuntimeError("DatabricksDurableApp supports one entrypoint")
        self._entrypoint = function
        return function

    async def __call__(self, scope, receive, send) -> None:
        await self.asgi_app(scope, receive, send)

    async def _execute(
        self,
        request: JsonObject,
        execution_context: DurableExecutionContext,
    ) -> JsonObject:
        if self._entrypoint is None:
            raise RuntimeError("register an @app.entrypoint before serving requests")
        session_id = request.get("session_id")
        payload = request.get("payload")
        if not isinstance(session_id, str) or not isinstance(payload, dict):
            raise TypeError("persisted request must contain session_id and payload")
        context = DurableAgentContext(
            run_id=execution_context.execution_id,
            session_id=session_id,
            attempt=execution_context.attempt,
            _execution_context=execution_context,
        )
        return await self._entrypoint(copy.deepcopy(payload), context)

    async def _invoke(self, request: Request):
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, "request body must be a JSON object")

        payload = copy.deepcopy(payload)
        is_background = bool(payload.pop(_BACKGROUND_KEY, False))
        is_stream = bool(payload.pop(_STREAM_KEY, False))
        run_id = request.headers.get(RUN_ID_HEADER) or f"inv_{uuid.uuid4().hex[:24]}"
        session_id = request.headers.get(SESSION_ID_HEADER) or str(uuid.uuid4())
        persisted_request = {"session_id": session_id, "payload": payload}
        headers = self._context_headers(run_id, session_id)

        try:
            if is_background:
                state = await self.runtime.submit(run_id, persisted_request)
                status_code = 200 if state.is_terminal else 202
                return JSONResponse(
                    self._state_payload(state), status_code=status_code, headers=headers
                )
            if is_stream:
                await self.runtime.submit(run_id, persisted_request)
                return StreamingResponse(
                    self._event_stream(run_id),
                    media_type="text/event-stream",
                    headers=headers,
                )
            response = await self.runtime.invoke(run_id, persisted_request)
            return JSONResponse(response, headers=headers)
        except DurableRequestConflictError as exc:
            raise HTTPException(
                409, "idempotency key was already used for another request"
            ) from exc
        except DurableExecutionFailedError as exc:
            raise HTTPException(500, "agent execution failed") from exc

    async def _get(self, run_id: str) -> JSONResponse:
        state = await self.runtime.get(run_id)
        if state is None:
            raise HTTPException(404, "run not found")
        session_id = str(state.request.get("session_id", ""))
        return JSONResponse(
            self._state_payload(state),
            headers=self._context_headers(run_id, session_id),
        )

    async def _events(self, run_id: str, after: int = 0) -> StreamingResponse:
        state = await self.runtime.get(run_id)
        if state is None:
            raise HTTPException(404, "run not found")
        session_id = str(state.request.get("session_id", ""))
        return StreamingResponse(
            self._event_stream(run_id, after),
            media_type="text/event-stream",
            headers=self._context_headers(run_id, session_id),
        )

    async def _event_stream(self, run_id: str, after: int = 0) -> AsyncIterator[str]:
        cursor = after
        while True:
            events = await self.runtime.events(run_id, after_sequence=cursor)
            for event in events:
                cursor = event.sequence_number
                yield f"id: {cursor}\ndata: {json.dumps(event.event)}\n\n"

            state = await self.runtime.get(run_id)
            if state is None:
                return
            if state.status == DurableExecutionStatus.FAILED:
                yield f"data: {json.dumps({'error': 'agent execution failed'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            if state.status == DurableExecutionStatus.COMPLETED:
                yield "data: [DONE]\n\n"
                return
            await asyncio.sleep(self.runtime.poll_seconds)

    @staticmethod
    def _state_payload(state: DurableExecution) -> JsonObject:
        if state.status == DurableExecutionStatus.COMPLETED:
            return {
                "id": state.execution_id,
                "status": "completed",
                **copy.deepcopy(state.response or {}),
            }
        if state.status == DurableExecutionStatus.FAILED:
            return {
                "id": state.execution_id,
                "status": "failed",
                "error": "agent execution failed",
            }
        return {"id": state.execution_id, "status": "in_progress"}

    @staticmethod
    def _context_headers(run_id: str, session_id: str) -> dict[str, str]:
        headers = {RUN_ID_HEADER: run_id}
        if session_id:
            headers[SESSION_ID_HEADER] = session_id
        return headers

    @staticmethod
    def _default_store(
        *,
        autoscaling_endpoint: str | None,
        project: str | None,
        branch: str | None,
        workspace_client: WorkspaceClient | None,
        schema: str,
    ) -> DurabilityStore:
        has_lakebase_config = bool(
            autoscaling_endpoint
            or project
            or branch
            or os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT")
            or os.getenv("LAKEBASE_AUTOSCALING_PROJECT")
            or os.getenv("LAKEBASE_AUTOSCALING_BRANCH")
            or os.getenv(_SESSION_STORE_ENV)
        )
        if not has_lakebase_config:
            return InMemoryDurabilityStore()
        if (
            os.getenv(_SESSION_STORE_ENV)
            and not autoscaling_endpoint
            and not project
            and not branch
            and not os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT")
            and not os.getenv("LAKEBASE_AUTOSCALING_PROJECT")
            and not os.getenv("LAKEBASE_AUTOSCALING_BRANCH")
        ):
            project = _SHARED_SESSION_STORE_PROJECT
            branch = _DEFAULT_SESSION_STORE_BRANCH
        return LakebaseDurabilityStore(
            autoscaling_endpoint=autoscaling_endpoint,
            project=project,
            branch=branch,
            workspace_client=workspace_client,
            schema=schema,
        )

    @staticmethod
    async def _health() -> JsonObject:
        return {"status": "ok"}
