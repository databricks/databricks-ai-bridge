"""FastAPI adapter for the Agent Bricks agent runtime."""

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

from databricks_agentbricks.runtime.auth import AuthError, InvocationAuthPolicy, RequestAuthContext
from databricks_agentbricks.runtime.runtime import Runtime
from databricks_agentbricks.runtime.store import RuntimeStore
from databricks_agentbricks.runtime.types import (
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


class DurableAgentServer(FastAPI):
    """Expose agent handlers through the Agent Bricks invocation HTTP protocol.

    ``ab dev`` selects a process-local Runtime Store. A deployed Agent Bricks server receives a
    Lakebase-backed Runtime Store, which preserves invocation state and can recover stale work when
    a handler is registered with :meth:`recover`.
    """

    def __init__(
        self,
        *,
        runtime_store: RuntimeStore | None = None,
        auth_policy: InvocationAuthPolicy | None = None,
    ) -> None:
        self.auth_policy = (
            auth_policy
            if auth_policy is not None
            else InvocationAuthPolicy.from_manifest(allow_missing=True)
        )
        self._invoke_hook: InvocationHook | None = None
        self._recovery_hook: InvocationHook | None = None
        self._request_auth: dict[str, RequestAuthContext] = {}
        if runtime_store is None:
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
            if self._runtime.is_durable and self._recovery_hook is None:
                logger.warning(
                    "No @app.recover handler is registered; automatic crash recovery is disabled."
                )
            await self._runtime.start()
            try:
                yield
            finally:
                await self._runtime.stop()
                self._close_request_auth()

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
        invocation_id = attempt_context.invocation_id
        request_auth = None
        if self.auth_policy.requires_user:
            request_auth = self._request_auth.pop(attempt_context.invocation_id, None)
            if attempt_context.is_recovery:
                if request_auth is not None:
                    request_auth.close()
                raise AuthError(
                    "MCP_USER_AUTH_RECOVERY_UNSUPPORTED",
                    "Request-user execution does not survive failure recovery yet",
                    500,
                )
            if request_auth is None:
                raise AuthError(
                    "MCP_USER_AUTH_RECOVERY_UNSUPPORTED",
                    "Request-user execution does not survive failure recovery yet",
                    500,
                )
            invocation_id = invocation_request.get("invocation_id")
            if not isinstance(invocation_id, str):
                request_auth.close()
                raise TypeError("request-user invocation must contain invocation_id")

        context = InvocationContext(
            invocation_id=invocation_id,
            session_id=session_id,
            attempt=attempt_context.attempt,
            _attempt_context=attempt_context,
            request_auth=request_auth,
        )
        function = self._recovery_hook if context.is_recovery else self._invoke_hook
        try:
            if function is None:
                handler = "@app.recover" if context.is_recovery else "@app.invoke"
                raise RuntimeError(f"no {handler} handler is registered")
            return await function(copy.deepcopy(invocation_request["input"]), context)
        finally:
            if request_auth is not None:
                request_auth.close()

    async def _invoke_request(self, request: Request, body: _InvocationRequest) -> Response:
        invocation_id = str(body.id)
        runtime_invocation_id = invocation_id
        request_auth = None
        registered_auth = False
        execution_owns_auth = False
        session_id = request.state.session_id or invocation_id
        if self.auth_policy.requires_user:
            request_auth = RequestAuthContext.from_headers(request.headers)
            runtime_invocation_id = request_auth.namespace("invocation", invocation_id)
            session_id = request_auth.namespace("session", session_id)
            existing_auth = self._request_auth.setdefault(runtime_invocation_id, request_auth)
            registered_auth = existing_auth is request_auth
            if not registered_auth:
                request_auth.close()
        invocation_request: JsonObject = {
            "session_id": session_id,
            "input": copy.deepcopy(body.input),
        }
        if self.auth_policy.requires_user:
            invocation_request["invocation_id"] = invocation_id
        try:
            if body.background:
                state = await self._runtime.submit(runtime_invocation_id, invocation_request)
                execution_owns_auth = registered_auth and state.status == InvocationStatus.QUEUED
                return JSONResponse(
                    self._accepted_payload(state, stream=body.stream, invocation_id=invocation_id),
                    status_code=202,
                )
            if body.stream:
                state = await self._runtime.submit(runtime_invocation_id, invocation_request)
                execution_owns_auth = registered_auth and state.status == InvocationStatus.QUEUED
                return StreamingResponse(
                    self._event_stream(runtime_invocation_id),
                    media_type="text/event-stream",
                )
            output = await self._runtime.invoke(runtime_invocation_id, invocation_request)
            return JSONResponse({"id": invocation_id, "status": "completed", "output": output})
        except InvocationConflictError as exc:
            raise HTTPException(409, "id was already used for another request") from exc
        except InvocationFailedError as exc:
            raise HTTPException(500, "agent invocation failed") from exc
        finally:
            if registered_auth and not execution_owns_auth and request_auth is not None:
                self._discard_request_auth(runtime_invocation_id, request_auth)

    async def _auth_error(self, request: Request, error: Exception) -> JSONResponse:
        assert isinstance(error, AuthError)
        return JSONResponse({"error": error.payload()}, status_code=error.status_code)

    async def _get_request(self, request: Request, invocation_id: UUID) -> JSONResponse:
        public_invocation_id = str(invocation_id)
        runtime_invocation_id = self._runtime_invocation_id(request, public_invocation_id)
        state = await self._runtime.get_invocation(runtime_invocation_id)
        if state is None:
            raise HTTPException(404, "invocation not found")
        return JSONResponse(self._state_payload(state, invocation_id=public_invocation_id))

    async def _events(
        self, request: Request, invocation_id: UUID, after: int = 0
    ) -> StreamingResponse:
        public_invocation_id = str(invocation_id)
        runtime_invocation_id = self._runtime_invocation_id(request, public_invocation_id)
        if await self._runtime.get_invocation(runtime_invocation_id) is None:
            raise HTTPException(404, "invocation not found")
        return StreamingResponse(
            self._event_stream(runtime_invocation_id, after),
            media_type="text/event-stream",
        )

    async def _event_stream(self, invocation_id: str, after: int = 0) -> AsyncIterator[str]:
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
    def _accepted_payload(
        state: Invocation, *, stream: bool, invocation_id: str | None = None
    ) -> JsonObject:
        invocation_id = invocation_id or state.invocation_id
        payload: JsonObject = {
            "id": invocation_id,
            "status": state.status.value.lower(),
            "status_url": f"{_API_ROOT}/{invocation_id}",
        }
        if stream:
            payload["events_url"] = f"{_API_ROOT}/{invocation_id}/events"
        return payload

    @staticmethod
    def _state_payload(state: Invocation, *, invocation_id: str | None = None) -> JsonObject:
        payload: JsonObject = {
            "id": invocation_id or state.invocation_id,
            "status": state.status.value.lower(),
        }
        if state.status == InvocationStatus.COMPLETED:
            payload["output"] = copy.deepcopy(state.response)
        elif state.status == InvocationStatus.FAILED:
            payload["error"] = "agent invocation failed"
        return payload

    def _runtime_invocation_id(self, request: Request, invocation_id: str) -> str:
        if not self.auth_policy.requires_user:
            return invocation_id
        request_auth = RequestAuthContext.from_headers(request.headers)
        try:
            return request_auth.namespace("invocation", invocation_id)
        finally:
            request_auth.close()

    def _discard_request_auth(self, invocation_id: str, request_auth: RequestAuthContext) -> None:
        if self._request_auth.get(invocation_id) is request_auth:
            self._request_auth.pop(invocation_id)
        request_auth.close()

    def _close_request_auth(self) -> None:
        request_auths = list(self._request_auth.values())
        self._request_auth.clear()
        for request_auth in request_auths:
            request_auth.close()


# Deprecated compatibility alias. New code should use DurableAgentServer.
AgentApp = DurableAgentServer
