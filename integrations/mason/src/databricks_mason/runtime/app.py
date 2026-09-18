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

from databricks_mason.runtime.auth import AuthError, InvocationAuthPolicy, RequestAuthContext
from databricks_mason.runtime.runtime import Runtime
from databricks_mason.runtime.store import RuntimeStore
from databricks_mason.runtime.types import (
    Invocation,
    InvocationAttemptContext,
    InvocationConflictError,
    InvocationContext,
    InvocationFailedError,
    InvocationHook,
    InvocationStatus,
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

    ``mason dev`` selects a process-local Runtime Store. A deployed Mason server receives a
    Lakebase-backed Runtime Store, which preserves invocation state and can recover stale work when
    a handler is registered with :meth:`recover`.
    """

    def __init__(
        self,
        *,
        runtime_store: RuntimeStore | None = None,
        auth_policy: InvocationAuthPolicy | None = None,
    ) -> None:
        self.auth_policy = auth_policy or InvocationAuthPolicy()
        self._invoke_hook: InvocationHook | None = None
        self._recovery_hook: InvocationHook | None = None
        self._runtime: Runtime | None
        if self.auth_policy.requires_user:
            if runtime_store is not None:
                raise ValueError("Request-user execution cannot use an explicit Runtime Store")
            self._runtime = None
        elif runtime_store is None:
            self._runtime = Runtime.from_environment(
                self._execute,
                recovery_enabled=lambda: self._recovery_hook is not None,
            )
        else:
            self._runtime = Runtime.from_store(
                self._execute,
                runtime_store=runtime_store,
                recovery_enabled=lambda: self._recovery_hook is not None,
            )

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            if self._invoke_hook is None:
                raise RuntimeError("register an invocation handler with @app.invoke")
            if self.auth_policy.requires_user:
                yield
                return
            assert self._runtime is not None
            if self._runtime.is_durable and self._recovery_hook is None:
                logger.warning(
                    "No @app.recover handler is registered; automatic crash recovery is disabled."
                )
            await self._runtime.start()
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
        self.add_exception_handler(AuthError, self._auth_error)
        self.middleware("http")(self._bind_session)
        self.add_api_route(_API_ROOT, self._invoke_request, methods=["POST"])
        self.add_api_route(f"{_API_ROOT}/{{invocation_id}}", self._get_request, methods=["GET"])
        self.add_api_route(
            f"{_API_ROOT}/{{invocation_id}}/events",
            self._events,
            methods=["GET"],
        )

    def invoke(self, function: InvocationHook) -> InvocationHook:
        """Register the handler for an invocation's first attempt."""
        if self._invoke_hook is not None:
            raise ValueError("an invocation handler is already registered")
        self._invoke_hook = function
        return function

    def recover(self, function: InvocationHook) -> InvocationHook:
        """Register the handler used after an interrupted attempt becomes stale."""
        if self.auth_policy.requires_user:
            raise ValueError("Request-user execution does not support durable recovery")
        if self._recovery_hook is not None:
            raise ValueError("a recovery handler is already registered")
        self._recovery_hook = function
        return function

    async def _bind_session(self, request: Request, call_next) -> Response:
        # TODO: Read the standard session header once Databricks Apps supports one. The Apps proxy
        # currently consumes its routing cookie before forwarding deployed requests, so the
        # invocation ID becomes the deterministic session fallback in _invoke_request.
        request.state.session_id = request.cookies.get(_ROUTING_COOKIE)
        return await call_next(request)

    async def _execute(
        self,
        invocation_request: JsonValue,
        attempt_context: InvocationAttemptContext,
    ) -> JsonValue:
        if not isinstance(invocation_request, dict):
            raise TypeError("invocation request must be an object")
        session_id = invocation_request.get("session_id")
        if not isinstance(session_id, str) or "input" not in invocation_request:
            raise TypeError("invocation request must contain session_id and input")

        context = InvocationContext(
            invocation_id=attempt_context.invocation_id,
            session_id=session_id,
            attempt=attempt_context.attempt,
            _attempt_context=attempt_context,
        )
        function = self._recovery_hook if context.is_recovery else self._invoke_hook
        if function is None:
            handler = "@app.recover" if context.is_recovery else "@app.invoke"
            raise RuntimeError(f"no {handler} handler is registered")
        return await function(copy.deepcopy(invocation_request["input"]), context)

    async def _invoke_request_user(
        self, request: Request, body: _InvocationRequest, invocation_id: str
    ) -> Response:
        if body.background:
            raise AuthError(
                "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
                "Request-user tools require foreground execution",
                400,
            )
        auth = RequestAuthContext.from_headers(request.headers)
        stream_owns_auth = False
        try:
            if self._invoke_hook is None:
                raise RuntimeError("no @app.invoke handler is registered")
            session_id = auth.namespace("session", request.state.session_id or invocation_id)
            if body.stream:
                response = StreamingResponse(
                    self._request_user_event_stream(body, invocation_id, session_id, auth),
                    media_type="text/event-stream",
                )
                stream_owns_auth = True
                return response

            sequence = 0

            async def discard_event(event: JsonObject) -> int:
                nonlocal sequence
                sequence += 1
                return sequence

            context = InvocationContext(
                invocation_id=invocation_id,
                session_id=session_id,
                attempt=1,
                _attempt_context=InvocationAttemptContext(invocation_id, 1, discard_event),
                request_auth=auth,
            )
            output = await self._invoke_hook(copy.deepcopy(body.input), context)
            return JSONResponse({"id": invocation_id, "status": "completed", "output": output})
        except AuthError:
            raise
        except Exception as exc:
            raise HTTPException(500, "agent invocation failed") from exc
        finally:
            if not stream_owns_auth:
                auth.close()

    async def _request_user_event_stream(
        self,
        body: _InvocationRequest,
        invocation_id: str,
        session_id: str,
        auth: RequestAuthContext,
    ) -> AsyncIterator[str]:
        events: asyncio.Queue[tuple[int, JsonObject] | None] = asyncio.Queue(maxsize=1)
        sequence = 0

        async def emit(event: JsonObject) -> int:
            nonlocal sequence
            sequence += 1
            await events.put((sequence, copy.deepcopy(event)))
            return sequence

        context = InvocationContext(
            invocation_id=invocation_id,
            session_id=session_id,
            attempt=1,
            _attempt_context=InvocationAttemptContext(invocation_id, 1, emit),
            request_auth=auth,
        )

        async def execute() -> None:
            await emit({"type": "run.started"})
            try:
                assert self._invoke_hook is not None
                await self._invoke_hook(copy.deepcopy(body.input), context)
            except asyncio.CancelledError:
                raise
            except AuthError as exc:
                failure: JsonObject = {
                    "type": "run.failed",
                    "error": str(exc),
                    "code": exc.code,
                }
                if exc.integration_id is not None:
                    failure["integration_id"] = exc.integration_id
                await emit(failure)
            except Exception:
                logger.error("Request-user streaming invocation failed: %s", invocation_id)
                await emit({"type": "run.failed", "error": "agent invocation failed"})
            else:
                await emit({"type": "run.completed"})
            await events.put(None)

        task = asyncio.create_task(execute(), name=f"request-user-{invocation_id}")
        try:
            while True:
                item = await events.get()
                if item is None:
                    await task
                    return
                sequence_number, event = item
                event_type = event.get("type", "message")
                yield (f"id: {sequence_number}\nevent: {event_type}\ndata: {json.dumps(event)}\n\n")
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            auth.close()

    async def _invoke_request(self, request: Request, body: _InvocationRequest) -> Response:
        invocation_id = str(body.id)
        invocation_request: JsonObject = {
            "session_id": request.state.session_id or invocation_id,
            "input": copy.deepcopy(body.input),
        }
        try:
            if self.auth_policy.requires_user:
                return await self._invoke_request_user(request, body, invocation_id)
            assert self._runtime is not None
            if body.background:
                state = await self._runtime.submit(invocation_id, invocation_request)
                return JSONResponse(
                    self._accepted_payload(state, stream=body.stream),
                    status_code=202,
                )
            if body.stream:
                await self._runtime.submit(invocation_id, invocation_request)
                return StreamingResponse(
                    self._event_stream(invocation_id),
                    media_type="text/event-stream",
                )
            output = await self._runtime.invoke(invocation_id, invocation_request)
            return JSONResponse({"id": invocation_id, "status": "completed", "output": output})
        except InvocationConflictError as exc:
            raise HTTPException(409, "id was already used for another request") from exc
        except InvocationFailedError as exc:
            raise HTTPException(500, "agent invocation failed") from exc

    async def _auth_error(self, request: Request, error: Exception) -> JSONResponse:
        assert isinstance(error, AuthError)
        return JSONResponse({"error": error.payload()}, status_code=error.status_code)

    async def _get_request(self, request: Request, invocation_id: UUID) -> JSONResponse:
        if self.auth_policy.requires_user:
            raise HTTPException(404, "invocation not found")
        assert self._runtime is not None
        state = await self._runtime.get_invocation(str(invocation_id))
        if state is None:
            raise HTTPException(404, "invocation not found")
        return JSONResponse(self._state_payload(state))

    async def _events(
        self, request: Request, invocation_id: UUID, after: int = 0
    ) -> StreamingResponse:
        normalized_invocation_id = str(invocation_id)
        if self.auth_policy.requires_user:
            raise HTTPException(404, "invocation not found")
        assert self._runtime is not None
        if await self._runtime.get_invocation(normalized_invocation_id) is None:
            raise HTTPException(404, "invocation not found")
        return StreamingResponse(
            self._event_stream(normalized_invocation_id, after),
            media_type="text/event-stream",
        )

    async def _event_stream(self, invocation_id: str, after: int = 0) -> AsyncIterator[str]:
        assert self._runtime is not None
        cursor = after
        while True:
            for event in await self._runtime.get_events(invocation_id, after_sequence=cursor):
                cursor = event.sequence_number
                event_type = event.event.get("type", "message")
                yield f"id: {cursor}\nevent: {event_type}\ndata: {json.dumps(event.event)}\n\n"

            state = await self._runtime.get_invocation(invocation_id)
            if state is None or state.is_terminal:
                return
            await asyncio.sleep(self._runtime.poll_seconds)

    @staticmethod
    def _accepted_payload(state: Invocation, *, stream: bool) -> JsonObject:
        invocation_id = state.invocation_id
        payload: JsonObject = {
            "id": invocation_id,
            "status": state.status.value.lower(),
            "status_url": f"{_API_ROOT}/{invocation_id}",
        }
        if stream:
            payload["events_url"] = f"{_API_ROOT}/{invocation_id}/events"
        return payload

    @staticmethod
    def _state_payload(state: Invocation) -> JsonObject:
        payload: JsonObject = {
            "id": state.invocation_id,
            "status": state.status.value.lower(),
        }
        if state.status == InvocationStatus.COMPLETED:
            payload["output"] = copy.deepcopy(state.response)
        elif state.status == InvocationStatus.FAILED:
            payload["error"] = "agent invocation failed"
        return payload
