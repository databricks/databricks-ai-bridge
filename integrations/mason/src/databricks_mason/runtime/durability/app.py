"""FastAPI adapter for Mason's agent runtime."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic import JsonValue as PydanticJsonValue

from databricks_mason.runtime.durability.runtime import DurableRuntime
from databricks_mason.runtime.durability.store import (
    InMemoryDurabilityStore,
    default_durability_store,
)
from databricks_mason.runtime.durability.types import (
    DurabilityStore,
    DurableAgentContext,
    DurableAgentHook,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
    JsonValue,
)

logger = logging.getLogger(__name__)

_ROUTING_COOKIE = "__Host-databricks-app-router"
_API_ROOT = "/api/invocations"


class _InvocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    input: PydanticJsonValue = Field(default_factory=list)
    background: bool = False
    stream: bool = False


class AgentApp(FastAPI):
    """Expose agent handlers through Mason's invocation HTTP protocol.

    The default runtime keeps background state and events in this process. Set
    ``durable_runtime=True`` to use Mason's deployed Lakebase store, heartbeats, and crash recovery.
    """

    def __init__(
        self,
        *,
        durable_runtime: bool = False,
        durability_store: DurabilityStore | None = None,
    ) -> None:
        self.durable_runtime = durable_runtime
        self._invoke_hook: DurableAgentHook | None = None
        self._on_recovery_hook: DurableAgentHook | None = None
        store = durability_store
        if store is None:
            store = default_durability_store() if durable_runtime else InMemoryDurabilityStore()
        self._runtime = DurableRuntime(
            self._execute,
            durability_store=store,
        )

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            if self._invoke_hook is None:
                raise RuntimeError("register an invocation handler with @app.invoke")
            recover = self.durable_runtime and self._on_recovery_hook is not None
            if self.durable_runtime and not recover:
                logger.warning(
                    "No @app.on_recovery handler is registered; automatic crash recovery is "
                    "disabled."
                )
            await self._runtime.start(recover=recover)
            try:
                yield
            finally:
                await self._runtime.stop()

        super().__init__(
            title="Databricks Agent Runtime",
            lifespan=lifespan,
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
        self.middleware("http")(self._bind_session)
        self.add_api_route(_API_ROOT, self._invoke_request, methods=["POST"])
        self.add_api_route(f"{_API_ROOT}/{{invocation_id}}", self._get_request, methods=["GET"])
        self.add_api_route(
            f"{_API_ROOT}/{{invocation_id}}/events",
            self._events,
            methods=["GET"],
        )

    def invoke(self, function: DurableAgentHook) -> DurableAgentHook:
        """Register the handler for an invocation's first attempt."""
        if self._invoke_hook is not None:
            raise ValueError("an invocation handler is already registered")
        self._invoke_hook = function
        return function

    def on_recovery(self, function: DurableAgentHook) -> DurableAgentHook:
        """Register the handler used after an interrupted attempt becomes stale."""
        if self._on_recovery_hook is not None:
            raise ValueError("a recovery handler is already registered")
        self._on_recovery_hook = function
        return function

    async def _bind_session(self, request: Request, call_next) -> Response:
        # TODO: Read the standard session header once Databricks Apps supports one. The Apps proxy
        # currently consumes its routing cookie before forwarding deployed requests, so the
        # invocation ID becomes the deterministic session fallback in _invoke_request.
        request.state.session_id = request.cookies.get(_ROUTING_COOKIE)
        return await call_next(request)

    async def _execute(
        self,
        execution_request: JsonValue,
        execution_context: DurableExecutionContext,
    ) -> JsonValue:
        if not isinstance(execution_request, dict):
            raise TypeError("execution request must be an object")
        session_id = execution_request.get("session_id")
        if not isinstance(session_id, str) or "input" not in execution_request:
            raise TypeError("execution request must contain session_id and input")

        context = DurableAgentContext(
            invocation_id=execution_context.execution_id,
            session_id=session_id,
            attempt=execution_context.attempt,
            _execution_context=execution_context,
        )
        function = self._on_recovery_hook if context.is_recovery else self._invoke_hook
        if function is None:
            handler = "@app.on_recovery" if context.is_recovery else "@app.invoke"
            raise RuntimeError(f"no {handler} handler is registered")
        return await function(copy.deepcopy(execution_request["input"]), context)

    async def _invoke_request(self, request: Request, body: _InvocationRequest) -> Response:
        invocation_id = str(body.id)
        execution_request: JsonObject = {
            "session_id": request.state.session_id or invocation_id,
            "input": copy.deepcopy(body.input),
        }
        try:
            if body.background:
                state = await self._runtime.submit(invocation_id, execution_request)
                return JSONResponse(
                    self._accepted_payload(state, stream=body.stream),
                    status_code=202,
                )
            if body.stream:
                await self._runtime.submit(invocation_id, execution_request)
                return StreamingResponse(
                    self._event_stream(invocation_id),
                    media_type="text/event-stream",
                )
            output = await self._runtime.invoke(invocation_id, execution_request)
            return JSONResponse({"id": invocation_id, "status": "completed", "output": output})
        except DurableRequestConflictError as exc:
            raise HTTPException(409, "id was already used for another request") from exc
        except DurableExecutionFailedError as exc:
            raise HTTPException(500, "agent execution failed") from exc

    async def _get_request(self, invocation_id: UUID) -> JSONResponse:
        state = await self._runtime.get_execution(str(invocation_id))
        if state is None:
            raise HTTPException(404, "invocation not found")
        return JSONResponse(self._state_payload(state))

    async def _events(self, invocation_id: UUID, after: int = 0) -> StreamingResponse:
        normalized_invocation_id = str(invocation_id)
        if await self._runtime.get_execution(normalized_invocation_id) is None:
            raise HTTPException(404, "invocation not found")
        return StreamingResponse(
            self._event_stream(normalized_invocation_id, after),
            media_type="text/event-stream",
        )

    async def _event_stream(self, invocation_id: str, after: int = 0) -> AsyncIterator[str]:
        cursor = after
        while True:
            for event in await self._runtime.get_events(invocation_id, after_sequence=cursor):
                cursor = event.sequence_number
                event_type = event.event.get("type", "message")
                yield f"id: {cursor}\nevent: {event_type}\ndata: {json.dumps(event.event)}\n\n"

            state = await self._runtime.get_execution(invocation_id)
            if state is None or state.is_terminal:
                return
            await asyncio.sleep(self._runtime.poll_seconds)

    @staticmethod
    def _accepted_payload(state: DurableExecution, *, stream: bool) -> JsonObject:
        invocation_id = state.execution_id
        payload: JsonObject = {
            "id": invocation_id,
            "status": state.status.value.lower(),
            "status_url": f"{_API_ROOT}/{invocation_id}",
        }
        if stream:
            payload["events_url"] = f"{_API_ROOT}/{invocation_id}/events"
        return payload

    @staticmethod
    def _state_payload(state: DurableExecution) -> JsonObject:
        payload: JsonObject = {
            "id": state.execution_id,
            "status": state.status.value.lower(),
        }
        if state.status == DurableExecutionStatus.COMPLETED:
            payload["output"] = copy.deepcopy(state.response)
        elif state.status == DurableExecutionStatus.FAILED:
            payload["error"] = "agent execution failed"
        return payload
