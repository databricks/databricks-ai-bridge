"""FastAPI adapter for the Agent Bricks agent runtime."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic import JsonValue as PydanticJsonValue
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from databricks_agentkit.runtime.auth import AuthError, InvocationAuthPolicy, RequestAuthContext
from databricks_agentkit.runtime.cors import (
    allowed_origins_from_environment,
    validate_allowed_origins,
)
from databricks_agentkit.runtime.errors import InvocationError, describe_failure
from databricks_agentkit.runtime.runtime import Runtime
from databricks_agentkit.runtime.store import RuntimeStore
from databricks_agentkit.runtime.types import (
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

_API_ROOT = "/api/invocations"
# Failed rows stored before failures were described carry no error object.
_UNKNOWN_FAILURE: JsonObject = {"code": "AGENT_ERROR", "message": "agent invocation failed"}


class _InvocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    session_id: str | None = Field(default=None, min_length=1)
    input: PydanticJsonValue = Field(default_factory=list)
    background: bool = False
    stream: bool = False


class DurableAgentServer(FastAPI):
    """Expose agent handlers through the Agent Bricks invocation HTTP protocol.

    ``agentbricks dev`` selects a process-local Runtime Store. A deployed Agent Bricks server receives a
    Lakebase-backed Runtime Store, which preserves invocation state and can recover stale work when
    a handler is registered with :meth:`recover`.
    """

    def __init__(
        self,
        *,
        runtime_store: RuntimeStore | None = None,
        auth_policy: InvocationAuthPolicy | None = None,
        allowed_origins: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._cors_allowed_origins = (
            allowed_origins_from_environment()
            if allowed_origins is None
            else validate_allowed_origins(allowed_origins)
        )
        self.auth_policy = (
            auth_policy
            if auth_policy is not None
            else InvocationAuthPolicy.from_manifest(allow_missing=True)
        )
        self._invoke_hook: InvocationHook | None = None
        self._recovery_hook: InvocationHook | None = None
        self._request_auth: dict[str, RequestAuthContext] = {}
        self._pending_auth_invocations: set[str] = set()
        self._stream_updates: dict[str, set[asyncio.Event]] = {}
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
            auth_cleanup = (
                asyncio.create_task(self._cleanup_request_auth_loop())
                if self.auth_policy.requires_user
                else None
            )
            try:
                yield
            finally:
                if auth_cleanup is not None:
                    auth_cleanup.cancel()
                    await asyncio.gather(auth_cleanup, return_exceptions=True)
                try:
                    await self._runtime.stop()
                finally:
                    self._close_request_auth()

        super().__init__(
            title="Databricks Agent Runtime",
            lifespan=lifespan,
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
        self.add_exception_handler(AuthError, self._auth_error)
        self.add_api_route(_API_ROOT, self._invoke_request, methods=["POST"])
        self.add_api_route(f"{_API_ROOT}/{{invocation_id}}", self._get_request, methods=["GET"])
        self.add_api_route(
            f"{_API_ROOT}/{{invocation_id}}/events",
            self._events,
            methods=["GET"],
        )

    def build_middleware_stack(self) -> ASGIApp:
        app = super().build_middleware_stack()
        if not self._cors_allowed_origins:
            return app
        # Wrap ServerErrorMiddleware too so uncaught errors remain readable by approved frontends.
        return CORSMiddleware(
            app,
            allow_origins=list(self._cors_allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Routing-Key", "Last-Event-ID"],
            allow_credentials=False,
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

    async def _execute(
        self,
        invocation_request: JsonValue,
        attempt_context: InvocationAttemptContext,
    ) -> JsonValue:
        if not isinstance(invocation_request, dict):
            raise TypeError("invocation request must be an object")
        if "input" not in invocation_request:
            raise TypeError("invocation attempt must contain input")
        invocation_id = attempt_context.invocation_id
        request_auth = None
        if self.auth_policy.requires_user:
            request_auth = self._request_auth.pop(attempt_context.invocation_id, None)
            self._pending_auth_invocations.discard(attempt_context.invocation_id)
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

        async def emit(event: JsonObject) -> int:
            sequence = await attempt_context.emit(event)
            for reader in self._stream_updates.get(attempt_context.invocation_id, ()):
                reader.set()
            return sequence

        context = InvocationContext(
            invocation_id=invocation_id,
            session_id=attempt_context.session_id,
            attempt=attempt_context.attempt,
            _attempt_context=replace(attempt_context, _emit=emit),
            request_auth=request_auth,
        )
        function = self._recovery_hook if context.is_recovery else self._invoke_hook
        try:
            if function is None:
                handler = "@app.recover" if context.is_recovery else "@app.invoke"
                raise RuntimeError(f"no {handler} handler is registered")
            return await function(copy.deepcopy(invocation_request["input"]), context)
        except AuthError:
            raise
        except Exception as exc:
            if request_auth is None:
                raise
            # Describe the failure while the request token is still known, so it can be redacted.
            raise InvocationError(describe_failure(exc, request_auth.secrets())) from exc
        finally:
            if request_auth is not None:
                request_auth.close()

    async def _invoke_request(self, request: Request, body: _InvocationRequest) -> Response:
        invocation_id = str(body.id)
        runtime_invocation_id = invocation_id
        request_auth = None
        registered_auth = False
        execution_owns_auth = False
        session_id = body.session_id
        if self.auth_policy.requires_user:
            request_auth = RequestAuthContext.from_headers(request.headers)
            runtime_invocation_id = request_auth.namespace("invocation", invocation_id)
            if session_id is not None:
                session_id = request_auth.namespace("session", session_id)
            existing_auth = self._request_auth.setdefault(runtime_invocation_id, request_auth)
            registered_auth = existing_auth is request_auth
            if not registered_auth:
                request_auth.close()
        invocation_request: JsonObject = {
            "input": copy.deepcopy(body.input),
        }
        if self.auth_policy.requires_user:
            invocation_request["invocation_id"] = invocation_id
        try:
            state = await self._runtime.submit(
                runtime_invocation_id, invocation_request, session_id=session_id
            )
            # A retry can reach another replica while the original owner is still
            # queued. Only that owner should retain credentials for later execution.
            execution_owns_auth = (
                registered_auth
                and state.status == InvocationStatus.QUEUED
                and self._runtime.owns_invocation(state)
            )
            if (
                execution_owns_auth
                and self._request_auth.get(runtime_invocation_id) is request_auth
            ):
                self._pending_auth_invocations.add(runtime_invocation_id)
            if body.background:
                return JSONResponse(
                    self._accepted_payload(state, stream=body.stream, invocation_id=invocation_id),
                    status_code=202,
                )
            if body.stream:
                return StreamingResponse(
                    self._event_stream(runtime_invocation_id),
                    media_type="text/event-stream",
                )
            output = await self._runtime.wait(runtime_invocation_id)
            return JSONResponse({"id": invocation_id, "status": "completed", "output": output})
        except InvocationConflictError as exc:
            raise HTTPException(409, "id was already used for another request") from exc
        except InvocationFailedError:
            state = await self._runtime.get_invocation(runtime_invocation_id)
            error, status_code, from_agent = self._failure_response(state)
            if not from_agent:
                return JSONResponse({"error": error}, status_code=status_code)
            retry_after = error.get("retry_after")
            headers = (
                {"Retry-After": str(retry_after)}
                if status_code == 429 and isinstance(retry_after, int)
                else None
            )
            return JSONResponse(
                {"id": invocation_id, "status": "failed", "error": error},
                status_code=status_code,
                headers=headers,
            )
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

    async def _event_stream(self, invocation_id: str, after: int = 0) -> AsyncGenerator[str, None]:
        cursor = after
        changed = asyncio.Event()
        readers = self._stream_updates.setdefault(invocation_id, set())
        readers.add(changed)
        try:
            while True:
                # Register/clear before reading so a persisted delta cannot miss the waiter.
                changed.clear()
                # Read state first so a terminal snapshot's committed events are drained below.
                state = await self._runtime.get_invocation(invocation_id)
                for event in await self._runtime.get_events(invocation_id, after_sequence=cursor):
                    cursor = event.sequence_number
                    event_type = event.event.get("type", "message")
                    yield f"id: {cursor}\nevent: {event_type}\ndata: {json.dumps(event.event)}\n\n"

                if state is None or state.is_terminal:
                    return
                try:
                    # Other workers and terminal state changes retain the polling fallback.
                    await asyncio.wait_for(changed.wait(), timeout=self._runtime.poll_seconds)
                except asyncio.TimeoutError:
                    pass
        finally:
            readers.discard(changed)
            if not readers:
                self._stream_updates.pop(invocation_id, None)

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
            payload["error"] = DurableAgentServer._failure_response(state)[0]
        return payload

    @staticmethod
    def _failure_response(state: Invocation | None) -> tuple[JsonObject, int, bool]:
        """Return a failed invocation's error, HTTP status, and whether the agent raised it.

        Auth failures (stored without ``source``) keep their original ``{"error": ...}`` body.
        """
        fallback: tuple[JsonObject, int, bool] = (dict(_UNKNOWN_FAILURE), 500, True)
        if state is None or state.status != InvocationStatus.FAILED:
            return fallback
        if not isinstance(state.response, dict):
            return fallback
        error = state.response.get("error")
        status_code = state.response.get("status_code")
        if not isinstance(error, dict) or not isinstance(status_code, int):
            return fallback
        if not isinstance(error.get("code"), str) or not isinstance(error.get("message"), str):
            return fallback
        if status_code < 400 or status_code > 599:
            return fallback
        return copy.deepcopy(error), status_code, state.response.get("source") == "agent"

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
            self._pending_auth_invocations.discard(invocation_id)
        request_auth.close()

    async def _cleanup_request_auth_loop(self) -> None:
        """Release pending credentials when a different worker recovers their invocation."""
        while True:
            # Only inspect admitted requests: a newly registered context may still
            # be waiting for accept() to commit its row.
            for invocation_id in tuple(self._pending_auth_invocations):
                request_auth = self._request_auth.get(invocation_id)
                if request_auth is None:
                    continue
                try:
                    state = await self._runtime.runtime_store.get(invocation_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("Pending request authentication cleanup failed", exc_info=True)
                    continue
                if self._request_auth.get(invocation_id) is not request_auth:
                    continue  # Execution may have taken the context while the read was in flight.
                if (
                    state is None
                    or state.is_terminal
                    or state.attempt > 1
                    or not self._runtime.owns_invocation(state)
                ):
                    self._discard_request_auth(invocation_id, request_auth)
            await asyncio.sleep(self._runtime.poll_seconds)

    def _close_request_auth(self) -> None:
        request_auths = list(self._request_auth.values())
        self._request_auth.clear()
        self._pending_auth_invocations.clear()
        for request_auth in request_auths:
            request_auth.close()


# Deprecated compatibility alias. New code should use DurableAgentServer.
AgentApp = DurableAgentServer
