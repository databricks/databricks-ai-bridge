"""SDK-provided HTTP application for durable agent execution."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from databricks_mason.runtime.memory_store import InMemoryDurabilityStore
from databricks_mason.runtime.runtime import DurableRuntime
from databricks_mason.runtime.store import DEFAULT_DURABILITY_SCHEMA, LakebaseDurabilityStore
from databricks_mason.runtime.types import (
    DurabilityStore,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

RUN_ID_HEADER = "X-Databricks-Run-Id"
SESSION_ID_HEADER = "X-Databricks-Session-Id"

_RUN_ID_KEY = "run_id"
_SESSION_ID_KEY = "session_id"
_BACKGROUND_KEY = "background"
_STREAM_KEY = "stream"
_PAYLOAD_KEY = "payload"
_ACTOR_KEY = "actor"
RUNTIME_ENDPOINT_ENV = "DATABRICKS_MASON_RUNTIME_ENDPOINT"
_POSTGRES_ENV = ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER")
_ROUTING_COOKIE = "__Host-databricks-app-router"
_LOCAL_SESSION_COOKIE = "mason-local-session"
_USER_HEADERS = ("X-Forwarded-Email", "X-Forwarded-User")

AgentHook = Callable[[JsonObject, "DurableAgentContext"], Awaitable[JsonObject]]


@dataclass(frozen=True)
class DurableAgentContext:
    """Runtime context supplied to registered invoke and recovery hooks."""

    run_id: str
    session_id: str
    actor: str
    attempt: int
    _execution_context: DurableExecutionContext

    @property
    def is_recovery(self) -> bool:
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        """Persist an event and return its replay cursor."""
        return await self._execution_context.emit(event)


class DurableAgentApp:
    """Host framework-agnostic invoke and recovery hooks on a durable HTTP server.

    The application payload is the request body after the runtime-owned ``run_id``,
    ``session_id``, ``background``, and ``stream`` fields are removed. The runtime persists the
    payload, results, and emitted events for every invocation mode.
    """

    def __init__(
        self,
        *,
        invoke: AgentHook | None = None,
        recover: AgentHook | None = None,
        durability_store: DurabilityStore | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = DEFAULT_DURABILITY_SCHEMA,
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 0.1,
    ) -> None:
        self._invoke_hook = invoke
        self._recover_hook = recover
        store = durability_store or self._default_store(
            autoscaling_endpoint=autoscaling_endpoint,
            project=project,
            branch=branch,
            workspace_client=workspace_client,
            schema=schema,
        )
        self._runtime = DurableRuntime(
            self._execute,
            durability_store=store,
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

        self.asgi_app = FastAPI(title="Databricks Durable Agent", lifespan=lifespan)

        @self.asgi_app.middleware("http")
        async def bind_request_context(request: Request, call_next):
            routing_session = request.cookies.get(_ROUTING_COOKIE)
            local_session = request.cookies.get(_LOCAL_SESSION_COOKIE)
            request.state.session_id = routing_session or local_session or str(uuid.uuid4())
            response = await call_next(request)
            if not routing_session and not local_session:
                response.set_cookie(
                    _LOCAL_SESSION_COOKIE,
                    request.state.session_id,
                    httponly=True,
                    samesite="lax",
                )
            return response

        for prefix in ("", "/api"):
            self.asgi_app.add_api_route(
                f"{prefix}/invocations", self._create_invocation, methods=["POST"]
            )
            self.asgi_app.add_api_route(
                f"{prefix}/invocations/{{run_id}}", self._get_invocation, methods=["GET"]
            )
            self.asgi_app.add_api_route(
                f"{prefix}/invocations/{{run_id}}/events",
                self._get_events,
                methods=["GET"],
            )
            self.asgi_app.add_api_route(f"{prefix}/health", self._health, methods=["GET"])
        self.asgi_app.add_api_route("/api/session/new", self._new_session, methods=["POST"])
        self.asgi_app.add_api_route("/api/healthz", self._health, methods=["GET"])

    @property
    def is_durable(self) -> bool:
        """Whether invocation state survives process replacement."""
        return self._runtime.is_durable

    @property
    def heartbeat_seconds(self) -> float:
        """Heartbeat interval used by the internal execution runtime."""
        return self._runtime.heartbeat_seconds

    @property
    def stale_seconds(self) -> float:
        """Age after which another worker may recover an active invocation."""
        return self._runtime.stale_seconds

    def set_session(self, request: Request, response: Response, session_id: str) -> None:
        """Update the request session and its browser routing cookie."""
        request.state.session_id = session_id
        self._rotate_session_cookie(request, response, session_id)

    def invoke(self, function: AgentHook) -> AgentHook:
        """Register the hook used for the first attempt of each run."""
        if self._invoke_hook is not None:
            raise RuntimeError("DurableAgentApp supports one invoke hook")
        self._invoke_hook = function
        return function

    def recover(self, function: AgentHook) -> AgentHook:
        """Register the hook used after a process failure makes an attempt stale."""
        if self._recover_hook is not None:
            raise RuntimeError("DurableAgentApp supports one recovery hook")
        self._recover_hook = function
        return function

    async def __call__(self, scope, receive, send) -> None:
        await self.asgi_app(scope, receive, send)

    def run(self, *, host: str = "0.0.0.0", port: int | None = None) -> None:
        """Run the application with uvicorn."""
        import uvicorn

        resolved_port = port or int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", "8000")))
        uvicorn.run(self, host=host, port=resolved_port)

    async def _execute(
        self,
        request: JsonObject,
        execution_context: DurableExecutionContext,
    ) -> JsonObject:
        payload = request.get(_PAYLOAD_KEY)
        session_id = request.get(_SESSION_ID_KEY)
        actor = request.get(_ACTOR_KEY)
        if (
            not isinstance(payload, dict)
            or not isinstance(session_id, str)
            or not isinstance(actor, str)
        ):
            raise RuntimeError("persisted runtime request is invalid")

        hook = self._recover_hook if execution_context.is_recovery else self._invoke_hook
        if hook is None:
            hook_name = "recovery" if execution_context.is_recovery else "invoke"
            raise RuntimeError(f"no {hook_name} hook is registered")

        context = DurableAgentContext(
            run_id=execution_context.execution_id,
            session_id=session_id,
            actor=actor,
            attempt=execution_context.attempt,
            _execution_context=execution_context,
        )
        response = await hook(copy.deepcopy(payload), context)
        if not isinstance(response, dict):
            raise TypeError(f"agent hook must return a JSON object, got {type(response).__name__}")
        return response

    async def _create_invocation(self, request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(400, "request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(422, "request body must be a JSON object")

        payload = copy.deepcopy(body)
        run_id = self._runtime_identifier(payload.pop(_RUN_ID_KEY, None), prefix="inv_")
        payload.pop(_SESSION_ID_KEY, None)
        session_id = request.state.session_id
        is_background = self._runtime_boolean(payload.pop(_BACKGROUND_KEY, False), _BACKGROUND_KEY)
        is_stream = self._runtime_boolean(payload.pop(_STREAM_KEY, False), _STREAM_KEY)
        persisted_request = {
            _PAYLOAD_KEY: payload,
            _SESSION_ID_KEY: session_id,
            _ACTOR_KEY: self._request_actor(request),
        }
        headers = self._context_headers(run_id, session_id)

        try:
            if is_background:
                state = await self._runtime.submit(run_id, persisted_request)
                status_code = 200 if state.is_terminal else 202
                return JSONResponse(
                    self._state_payload(
                        state,
                        session_id=session_id,
                        events_url=self._events_url(request, run_id) if is_stream else None,
                    ),
                    status_code=status_code,
                    headers=headers,
                )
            if is_stream:
                await self._runtime.submit(run_id, persisted_request)
                return StreamingResponse(
                    self._event_stream(run_id),
                    media_type="text/event-stream",
                    headers={
                        **headers,
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            response = await self._runtime.invoke(run_id, persisted_request)
            return JSONResponse(response, headers=headers)
        except DurableRequestConflictError as exc:
            raise HTTPException(409, "run_id was already used for another request") from exc
        except DurableExecutionFailedError as exc:
            raise HTTPException(500, "agent execution failed") from exc

    async def _get_invocation(self, run_id: str) -> JSONResponse:
        state = await self._runtime.get(run_id)
        if state is None:
            raise HTTPException(404, "run not found")
        session_id = self._session_id(state)
        return JSONResponse(
            self._state_payload(state, session_id=session_id),
            headers=self._context_headers(run_id, session_id),
        )

    async def _get_events(self, run_id: str, after: int = 0) -> StreamingResponse:
        state = await self._runtime.get(run_id)
        if state is None:
            raise HTTPException(404, "run not found")
        session_id = self._session_id(state)
        return StreamingResponse(
            self._event_stream(run_id, after),
            media_type="text/event-stream",
            headers={
                **self._context_headers(run_id, session_id),
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _event_stream(self, run_id: str, after: int = 0) -> AsyncIterator[str]:
        cursor = after
        while True:
            events = await self._runtime.events(run_id, after_sequence=cursor)
            for event in events:
                cursor = event.sequence_number
                yield f"id: {cursor}\ndata: {json.dumps(event.event)}\n\n"

            state = await self._runtime.get(run_id)
            if state is None:
                return
            if state.status == DurableExecutionStatus.FAILED:
                yield f"data: {json.dumps({'error': 'agent execution failed'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            if state.status == DurableExecutionStatus.COMPLETED:
                yield "data: [DONE]\n\n"
                return
            await asyncio.sleep(self._runtime.poll_seconds)

    async def _new_session(self, request: Request) -> JSONResponse:
        previous_session_id = request.state.session_id
        session_id = str(uuid.uuid4())
        request.state.session_id = session_id
        response = JSONResponse(
            {"session_id": session_id, "previous_session_id": previous_session_id}
        )
        self.set_session(request, response, session_id)
        return response

    @classmethod
    def _state_payload(
        cls,
        state: DurableExecution,
        *,
        session_id: str,
        events_url: str | None = None,
    ) -> JsonObject:
        if state.status == DurableExecutionStatus.COMPLETED:
            return {
                "id": state.execution_id,
                "session_id": session_id,
                "status": "completed",
                "result": copy.deepcopy(state.response or {}),
            }
        if state.status == DurableExecutionStatus.FAILED:
            return {
                "id": state.execution_id,
                "session_id": session_id,
                "status": "failed",
                "error": "agent execution failed",
            }
        payload: JsonObject = {
            "id": state.execution_id,
            "session_id": session_id,
            "status": "in_progress",
        }
        if events_url is not None:
            payload["events_url"] = events_url
        return payload

    @staticmethod
    def _session_id(state: DurableExecution) -> str:
        session_id = state.request.get(_SESSION_ID_KEY)
        return session_id if isinstance(session_id, str) else ""

    @staticmethod
    def _runtime_identifier(
        value: Any,
        *,
        prefix: str = "",
        default: str | None = None,
    ) -> str:
        if value is None or value == "":
            return default or f"{prefix}{uuid.uuid4()}"
        if not isinstance(value, str):
            raise HTTPException(422, "run_id and session_id must be strings")
        return value

    @staticmethod
    def _runtime_boolean(value: Any, name: str) -> bool:
        if not isinstance(value, bool):
            raise HTTPException(422, f"{name} must be a boolean")
        return value

    @staticmethod
    def _events_url(request: Request, run_id: str) -> str:
        prefix = "/api" if request.url.path.startswith("/api/") else ""
        return f"{prefix}/invocations/{run_id}/events"

    @staticmethod
    def _context_headers(run_id: str, session_id: str) -> dict[str, str]:
        return {RUN_ID_HEADER: run_id, SESSION_ID_HEADER: session_id}

    @staticmethod
    def _request_actor(request: Request) -> str:
        for header in _USER_HEADERS:
            if actor := request.headers.get(header):
                return actor
        return "agent"

    @staticmethod
    def _rotate_session_cookie(request: Request, response: Response, session_id: str) -> None:
        if request.cookies.get(_ROUTING_COOKIE):
            response.set_cookie(
                _ROUTING_COOKIE,
                session_id,
                secure=True,
                httponly=True,
                samesite="lax",
                path="/",
            )
            response.delete_cookie(_LOCAL_SESSION_COOKIE, path="/")
        elif request.cookies.get(_LOCAL_SESSION_COOKIE):
            response.set_cookie(
                _LOCAL_SESSION_COOKIE,
                session_id,
                httponly=True,
                samesite="lax",
                path="/",
            )

    @staticmethod
    def _default_store(
        *,
        autoscaling_endpoint: str | None,
        project: str | None,
        branch: str | None,
        workspace_client: WorkspaceClient | None,
        schema: str,
    ) -> DurabilityStore:
        has_explicit_lakebase_config = bool(
            autoscaling_endpoint
            or project
            or branch
            or os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT")
            or os.getenv("LAKEBASE_AUTOSCALING_PROJECT")
            or os.getenv("LAKEBASE_AUTOSCALING_BRANCH")
        )
        if has_explicit_lakebase_config:
            return LakebaseDurabilityStore(
                autoscaling_endpoint=autoscaling_endpoint,
                project=project,
                branch=branch,
                workspace_client=workspace_client,
                schema=schema,
            )

        endpoint = os.getenv(RUNTIME_ENDPOINT_ENV)
        postgres = {name: os.getenv(name) for name in _POSTGRES_ENV}
        if endpoint:
            missing = [name for name, value in postgres.items() if not value]
            if missing:
                joined = ", ".join(missing)
                raise RuntimeError(
                    f"{RUNTIME_ENDPOINT_ENV} is set, but the Databricks Apps Postgres "
                    f"resource is missing: {joined}"
                )
            try:
                port = int(postgres["PGPORT"] or "")
            except ValueError as exc:
                raise RuntimeError("PGPORT must be an integer") from exc
            return LakebaseDurabilityStore.from_app_resource(
                endpoint=endpoint,
                host=postgres["PGHOST"] or "",
                port=port,
                database=postgres["PGDATABASE"] or "",
                username=postgres["PGUSER"] or "",
                sslmode=os.getenv("PGSSLMODE", "require"),
                workspace_client=workspace_client,
                schema=schema,
            )
        return InMemoryDurabilityStore()

    @staticmethod
    async def _health() -> JsonObject:
        return {"status": "ok"}
