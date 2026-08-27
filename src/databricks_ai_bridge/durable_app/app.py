"""ASGI application preserving the agent payload while adding durable metadata."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from databricks_ai_bridge.durable_runtime import (
    DatabricksDurableRuntime,
    DurabilityStore,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionStatus,
    JsonObject,
)

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
        return await self._execution_context.emit(event)


class DatabricksDurableApp:
    """Host one JSON entrypoint without wrapping the application's request body."""

    def __init__(
        self,
        *,
        durability_store: DurabilityStore | None = None,
        schema: str = "databricks_durable_app",
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
        path: str = "/invocations",
    ) -> None:
        self._entrypoint: DurableAgentEntrypoint | None = None
        self._resume_entrypoint: DurableAgentEntrypoint | None = None
        self._path = path.rstrip("/") or "/"
        self.runtime = DatabricksDurableRuntime(
            self._execute,
            durability_store=durability_store,
            schema=schema,
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
        self.asgi_app.add_api_route(self._path, self._submit, methods=["POST"])
        self.asgi_app.add_api_route(f"{self._path}/{{run_id}}", self._get, methods=["GET"])
        self.asgi_app.add_api_route(
            f"{self._path}/{{run_id}}/events",
            self._events,
            methods=["GET"],
        )
        self.asgi_app.add_api_route("/api/healthz", self._health, methods=["GET"])

    def entrypoint(self, function: DurableAgentEntrypoint) -> DurableAgentEntrypoint:
        """Register the single agent function invoked for every durable attempt."""
        if self._entrypoint is not None:
            raise RuntimeError("DatabricksDurableApp supports one entrypoint")
        self._entrypoint = function
        return function

    def on_resume(self, function: DurableAgentEntrypoint) -> DurableAgentEntrypoint:
        """Register the handler used after a stale attempt is reclaimed."""
        if self._resume_entrypoint is not None:
            raise RuntimeError("DatabricksDurableApp supports one resume entrypoint")
        self._resume_entrypoint = function
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
        function = (
            self._resume_entrypoint
            if context.is_recovery and self._resume_entrypoint is not None
            else self._entrypoint
        )
        return await function(copy.deepcopy(payload), context)

    async def _submit(self, http_request: Request) -> Response:
        payload = await http_request.json()
        if not isinstance(payload, dict):
            raise HTTPException(400, "request body must be a JSON object")

        run_id = http_request.headers.get("Idempotency-Key") or str(uuid4())
        session_id = http_request.headers.get("Databricks-Agent-Session-Id") or str(uuid4())
        background = self._header_bool(http_request, "Databricks-Background", False)
        stream = self._header_bool(http_request, "Databricks-Stream", False)
        request: JsonObject = {
            "session_id": session_id,
            "payload": payload,
        }
        response_headers = {
            "Databricks-Run-Id": run_id,
            "Databricks-Agent-Session-Id": session_id,
        }

        if stream:
            await self.runtime.submit(run_id, request)
            return StreamingResponse(
                self._event_stream(run_id, 0),
                media_type="text/event-stream",
                headers=response_headers,
            )

        if background:
            state = await self.runtime.submit(run_id, request)
            status_code = 200 if state.is_terminal else 202
            response_headers["Location"] = f"{self._path}/{run_id}"
            return JSONResponse(
                self._state_payload(state),
                status_code=status_code,
                headers=response_headers,
            )

        result = await self.runtime.invoke(run_id, request)
        return JSONResponse(result, headers=response_headers)

    async def _get(self, run_id: str) -> JSONResponse:
        state = await self.runtime.get(run_id)
        if state is None:
            raise HTTPException(404, "run not found")
        return JSONResponse(self._state_payload(state))

    async def _events(self, run_id: str, after: int = 0) -> StreamingResponse:
        if await self.runtime.get(run_id) is None:
            raise HTTPException(404, "run not found")
        return StreamingResponse(
            self._event_stream(run_id, after),
            media_type="text/event-stream",
        )

    async def _event_stream(self, run_id: str, after: int) -> AsyncIterator[str]:
        cursor = after
        while True:
            events = await self.runtime.events(run_id, after_sequence=cursor)
            for event in events:
                cursor = event.sequence_number
                yield (
                    f"id: {cursor}\n"
                    f"event: {event.event.get('type', 'message')}\n"
                    f"data: {json.dumps(event.event)}\n\n"
                )

            state = await self.runtime.get(run_id)
            if state is None or state.status in {
                DurableExecutionStatus.COMPLETED,
                DurableExecutionStatus.FAILED,
            }:
                return
            await asyncio.sleep(0.25)

    @staticmethod
    def _state_payload(state: DurableExecution) -> JsonObject:
        return {
            "run_id": state.execution_id,
            "status": state.status.value,
            "attempt": state.attempt,
            "result": copy.deepcopy(state.response),
        }

    @staticmethod
    async def _health() -> JsonObject:
        return {"status": "healthy"}

    @staticmethod
    def _header_bool(request: Request, name: str, default: bool) -> bool:
        value = request.headers.get(name)
        if value is None:
            return default
        normalized = value.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise HTTPException(400, f"{name} must be true or false")
