"""Thin FastAPI layer over Mason's durable runtime."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from databricks_mason.runtime.runtime import DurableRuntime
from databricks_mason.runtime.store import default_durability_store
from databricks_mason.runtime.types import (
    AgentHook,
    DurabilityStore,
    DurableAgentContext,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
)

ROUTING_COOKIE = "__Host-databricks-app-router"


class InvocationRequest(BaseModel):
    """The fixed HTTP request accepted by :class:`DurableAgentApp`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    input: Any = Field(default_factory=list)
    resume: JsonObject | None = None
    background: bool = False
    stream: bool = False

    def payload(self) -> JsonObject:
        payload: JsonObject = {"input": copy.deepcopy(self.input)}
        if "resume" in self.model_fields_set:
            payload["resume"] = copy.deepcopy(self.resume)
        return payload


class DurableAgentApp:
    """Serve one agent callback through Mason's durable HTTP protocol."""

    def __init__(
        self,
        invoke: AgentHook,
        *,
        on_resume: AgentHook | None = None,
        durability_store: DurabilityStore | None = None,
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 0.1,
    ) -> None:
        self._invoke = invoke
        self._on_resume = on_resume
        self._runtime = DurableRuntime(
            self._execute,
            durability_store=(
                durability_store if durability_store is not None else default_durability_store()
            ),
            heartbeat_seconds=heartbeat_seconds,
            stale_seconds=stale_seconds,
            scan_seconds=scan_seconds,
            poll_seconds=poll_seconds,
        )

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            await self._runtime.start()
            try:
                yield
            finally:
                await self._runtime.stop()

        self.app = FastAPI(title="Databricks Durable Agent", lifespan=lifespan)
        self.app.middleware("http")(self._bind_session)
        for prefix in ("", "/api"):
            self.app.add_api_route(f"{prefix}/invocations", self._invoke_request, methods=["POST"])
            self.app.add_api_route(
                f"{prefix}/invocations/{{run_id}}", self._get_request, methods=["GET"]
            )
            self.app.add_api_route(
                f"{prefix}/invocations/{{run_id}}/events", self._events, methods=["GET"]
            )
        self.app.add_api_route("/health", self._health, methods=["GET"])
        self.app.add_api_route("/api/healthz", self._health, methods=["GET"])

    async def _bind_session(self, request: Request, call_next) -> Response:
        session_id = request.cookies.get(ROUTING_COOKIE)
        if session_id is None:
            session_id = str(uuid.uuid4())
        request.state.session_id = session_id
        response = await call_next(request)
        if ROUTING_COOKIE not in request.cookies:
            response.set_cookie(
                ROUTING_COOKIE,
                session_id,
                secure=True,
                httponly=True,
                samesite="lax",
                path="/",
            )
        return response

    async def _execute(
        self,
        request: JsonObject,
        execution_context: DurableExecutionContext,
    ) -> JsonObject:
        payload = request.get("input")
        session_id = request.get("session_id")
        if not isinstance(payload, dict) or not isinstance(session_id, str):
            raise TypeError("persisted request must contain session_id and input")

        context = DurableAgentContext(
            run_id=execution_context.execution_id,
            session_id=session_id,
            attempt=execution_context.attempt,
            _execution_context=execution_context,
        )
        hook = self._on_resume if context.is_recovery and self._on_resume else self._invoke
        response = await hook(copy.deepcopy(payload), context)
        if not isinstance(response, dict):
            raise TypeError(
                f"agent callback must return a JSON object, got {type(response).__name__}"
            )
        return response

    async def _invoke_request(self, request: Request, body: InvocationRequest) -> Response:
        persisted_request: JsonObject = {
            "session_id": request.state.session_id,
            "input": body.payload(),
        }
        try:
            if body.stream:
                await self._runtime.submit(body.id, persisted_request)
                return StreamingResponse(
                    self._event_stream(body.id),
                    media_type="text/event-stream",
                )
            if body.background:
                state = await self._runtime.submit(body.id, persisted_request)
                return JSONResponse(
                    self._state_payload(state),
                    status_code=200 if state.is_terminal else 202,
                )
            return JSONResponse(await self._runtime.invoke(body.id, persisted_request))
        except DurableRequestConflictError as exc:
            raise HTTPException(409, "id was already used for another request") from exc
        except DurableExecutionFailedError as exc:
            raise HTTPException(500, "agent execution failed") from exc

    async def _get_request(self, run_id: str) -> JSONResponse:
        state = await self._runtime.get(run_id)
        if state is None:
            raise HTTPException(404, "run not found")
        return JSONResponse(self._state_payload(state))

    async def _events(self, run_id: str, after: int = 0) -> StreamingResponse:
        if await self._runtime.get(run_id) is None:
            raise HTTPException(404, "run not found")
        return StreamingResponse(
            self._event_stream(run_id, after),
            media_type="text/event-stream",
        )

    async def _event_stream(self, run_id: str, after: int = 0) -> AsyncIterator[str]:
        cursor = after
        while True:
            for event in await self._runtime.events(run_id, after_sequence=cursor):
                cursor = event.sequence_number
                event_type = event.event.get("type", "message")
                yield f"id: {cursor}\nevent: {event_type}\ndata: {json.dumps(event.event)}\n\n"

            state = await self._runtime.get(run_id)
            if state is None:
                return
            if state.status == DurableExecutionStatus.FAILED:
                yield f"data: {json.dumps({'error': 'agent execution failed'})}\n\n"
                return
            if state.status == DurableExecutionStatus.COMPLETED:
                return
            await asyncio.sleep(self._runtime.poll_seconds)

    @staticmethod
    def _state_payload(state: DurableExecution) -> JsonObject:
        payload: JsonObject = {
            "id": state.execution_id,
            "status": state.status.value.lower(),
            "attempt": state.attempt,
        }
        if state.status == DurableExecutionStatus.COMPLETED:
            payload.update(copy.deepcopy(state.response or {}))
        elif state.status == DurableExecutionStatus.FAILED:
            payload["error"] = "agent execution failed"
        return payload

    @staticmethod
    async def _health() -> JsonObject:
        return {"status": "ok"}
