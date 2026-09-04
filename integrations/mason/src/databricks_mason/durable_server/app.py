"""Thin HTTP server around the transport-neutral durable runtime."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from uuid_utils import uuid7

from databricks_mason.runtime.runtime import DurableRuntime
from databricks_mason.runtime.types import (
    DurabilityStore,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
)

RUN_ID_HEADER = "X-Databricks-Run-Id"
SESSION_ID_HEADER = "X-Databricks-Session-Id"

_ROUTING_COOKIE = "__Host-databricks-app-router"
_LOCAL_SESSION_COOKIE = "mason-local-session"
_USER_HEADERS = ("X-Forwarded-Email", "X-Forwarded-User")

AgentHook = Callable[[JsonObject, "DurableAgentContext"], Awaitable[JsonObject]]


@dataclass(frozen=True)
class DurableAgentContext:
    """Invocation metadata and durable event emission for an agent callback."""

    run_id: str
    session_id: str
    actor: str
    attempt: int
    _execution_context: DurableExecutionContext

    @property
    def is_recovery(self) -> bool:
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        return await self._execution_context.emit(event)


class DurableAgentApp:
    """Expose an agent callback through Mason's durable HTTP protocol."""

    def __init__(
        self,
        invoke: AgentHook,
        *,
        on_resume: AgentHook | None = None,
        durability_store: DurabilityStore,
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 0.1,
    ) -> None:
        self._invoke = invoke
        self._on_resume = on_resume
        self._runtime = DurableRuntime(
            self._execute,
            durability_store=durability_store,
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
        self.app.middleware("http")(self._bind_request_context)
        for prefix in ("", "/api"):
            self.app.add_api_route(
                f"{prefix}/invocations", self._create_invocation, methods=["POST"]
            )
            self.app.add_api_route(
                f"{prefix}/invocations/{{run_id}}", self._get_invocation, methods=["GET"]
            )
            self.app.add_api_route(
                f"{prefix}/invocations/{{run_id}}/events",
                self._get_events,
                methods=["GET"],
            )
            self.app.add_api_route(f"{prefix}/health", self._health, methods=["GET"])
        self.app.add_api_route("/api/session/new", self._new_session, methods=["POST"])
        self.app.add_api_route("/api/healthz", self._health, methods=["GET"])

    async def _bind_request_context(self, request: Request, call_next):
        routing_session = request.cookies.get(_ROUTING_COOKIE)
        local_session = request.cookies.get(_LOCAL_SESSION_COOKIE)
        request.state.session_id = routing_session or local_session or str(uuid7())
        response = await call_next(request)
        if not routing_session and not local_session:
            response.set_cookie(
                _LOCAL_SESSION_COOKIE,
                request.state.session_id,
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
        payload = request.get("payload")
        session_id = request.get("session_id")
        actor = request.get("actor")
        if (
            not isinstance(payload, dict)
            or not isinstance(session_id, str)
            or not isinstance(actor, str)
        ):
            raise RuntimeError("persisted runtime request is invalid")

        context = DurableAgentContext(
            run_id=execution_context.execution_id,
            session_id=session_id,
            actor=actor,
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

    async def _create_invocation(self, request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(400, "request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(422, "request body must be a JSON object")

        payload = copy.deepcopy(body)
        run_id = self._identifier(payload.pop("run_id", None), default=f"inv_{uuid7()}")
        payload.pop("session_id", None)
        payload.pop("actor", None)
        background = self._boolean(payload.pop("background", False), "background")
        stream = self._boolean(payload.pop("stream", False), "stream")
        session_id = request.state.session_id
        persisted_request = {
            "payload": payload,
            "session_id": session_id,
            "actor": self._request_actor(request),
        }
        headers = self._context_headers(run_id, session_id)

        try:
            if background:
                state = await self._runtime.submit(run_id, persisted_request)
                return JSONResponse(
                    self._state_payload(
                        state, session_id, self._events_url(request, run_id, stream)
                    ),
                    status_code=200 if state.is_terminal else 202,
                    headers=headers,
                )
            if stream:
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
            self._state_payload(state, session_id),
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
            for event in await self._runtime.events(run_id, after_sequence=cursor):
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
        session_id = str(uuid7())
        request.state.session_id = session_id
        response = JSONResponse(
            {"session_id": session_id, "previous_session_id": previous_session_id}
        )
        self._rotate_session_cookie(request, response, session_id)
        return response

    @staticmethod
    def _state_payload(
        state: DurableExecution,
        session_id: str,
        events_url: str | None = None,
    ) -> JsonObject:
        if state.status == DurableExecutionStatus.COMPLETED:
            payload = copy.deepcopy(state.response or {})
            payload.update(id=state.execution_id, session_id=session_id, status="completed")
            return payload
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
        if events_url:
            payload["events_url"] = events_url
        return payload

    @staticmethod
    def _identifier(value: Any, *, default: str) -> str:
        if value is None or value == "":
            return default
        if not isinstance(value, str):
            raise HTTPException(422, "run_id must be a string")
        return value

    @staticmethod
    def _boolean(value: Any, name: str) -> bool:
        if not isinstance(value, bool):
            raise HTTPException(422, f"{name} must be a boolean")
        return value

    @staticmethod
    def _session_id(state: DurableExecution) -> str:
        session_id = state.request.get("session_id")
        return session_id if isinstance(session_id, str) else ""

    @staticmethod
    def _events_url(request: Request, run_id: str, stream: bool) -> str | None:
        if not stream:
            return None
        prefix = "/api" if request.url.path.startswith("/api/") else ""
        return f"{prefix}/invocations/{run_id}/events"

    @staticmethod
    def _context_headers(run_id: str, session_id: str) -> dict[str, str]:
        return {RUN_ID_HEADER: run_id, SESSION_ID_HEADER: session_id}

    @staticmethod
    def _request_actor(request: Request) -> str:
        return next(
            (actor for header in _USER_HEADERS if (actor := request.headers.get(header))),
            "agent",
        )

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
    async def _health() -> JsonObject:
        return {"status": "ok"}
