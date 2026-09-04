"""A fixed JSON/HTTP protocol around the transport-neutral durable runtime."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from databricks_ai_bridge.durable_runtime import (
    DatabricksDurableRuntime,
    DurabilityStore,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionStatus,
    JsonObject,
)

DurableRequestHandler = Callable[[JsonObject, "DurableRequestContext"], Awaitable[JsonObject]]


@dataclass(frozen=True)
class DurableRequestContext:
    run_id: str
    session_id: str
    attempt: int
    _execution_context: DurableExecutionContext

    @property
    def is_recovery(self) -> bool:
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        return await self._execution_context.emit(event)


class InvocationRequest(BaseModel):
    id: str
    session_id: str
    input: dict[str, Any] = Field(default_factory=dict)
    background: bool = True
    stream: bool = False


class DatabricksDurableServer:
    """Serve one generic JSON handler through a standard durable protocol."""

    def __init__(
        self,
        handler: DurableRequestHandler,
        *,
        on_resume: DurableRequestHandler | None = None,
        durability_store: DurabilityStore | None = None,
        schema: str = "databricks_durable_server",
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
    ) -> None:
        self.handler = handler
        self.resume_handler = on_resume
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

        self.app = FastAPI(lifespan=lifespan)
        self.app.add_api_route("/invocations", self._invoke, methods=["POST"])
        self.app.add_api_route("/invocations/{run_id}", self._get, methods=["GET"])
        self.app.add_api_route(
            "/invocations/{run_id}/events",
            self._events,
            methods=["GET"],
        )
        self.app.add_api_route("/api/healthz", self._health, methods=["GET"])

    async def _execute(
        self,
        request: JsonObject,
        execution_context: DurableExecutionContext,
    ) -> JsonObject:
        session_id = request.get("session_id")
        payload = request.get("input")
        if not isinstance(session_id, str) or not isinstance(payload, dict):
            raise TypeError("persisted request must contain session_id and input")
        context = DurableRequestContext(
            run_id=execution_context.execution_id,
            session_id=session_id,
            attempt=execution_context.attempt,
            _execution_context=execution_context,
        )
        handler = (
            self.resume_handler
            if context.is_recovery and self.resume_handler is not None
            else self.handler
        )
        return await handler(copy.deepcopy(payload), context)

    async def _invoke(self, request: InvocationRequest) -> Response:
        persisted_request: JsonObject = {
            "session_id": request.session_id,
            "input": request.input,
        }
        if request.stream:
            await self.runtime.submit(request.id, persisted_request)
            return StreamingResponse(
                self._event_stream(request.id, 0),
                media_type="text/event-stream",
                headers={"Databricks-Run-Id": request.id},
            )

        if request.background:
            state = await self.runtime.submit(request.id, persisted_request)
            status_code = 200 if state.is_terminal else 202
            return JSONResponse(self._state_payload(state), status_code=status_code)

        response = await self.runtime.invoke(request.id, persisted_request)
        state = await self.runtime.get(request.id)
        if state is None:
            raise RuntimeError(f"execution {request.id!r} disappeared after completion")
        payload = self._state_payload(state)
        payload["output"] = response
        return JSONResponse(payload)

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
            "id": state.execution_id,
            "status": state.status.value.lower(),
            "attempt": state.attempt,
            "output": copy.deepcopy(state.response),
        }

    @staticmethod
    async def _health() -> JsonObject:
        return {"status": "healthy"}
