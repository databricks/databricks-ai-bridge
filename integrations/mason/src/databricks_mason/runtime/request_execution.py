"""Request-owned execution without a durable queue or recoverable credential state."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import AsyncIterator
from dataclasses import replace

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from databricks_mason.runtime.auth import AuthError, RequestAuthContext
from databricks_mason.runtime.store import InMemoryRuntimeStore
from databricks_mason.runtime.types import (
    Invocation,
    InvocationAttemptContext,
    InvocationConflictError,
    InvocationContext,
    InvocationEvent,
    InvocationHook,
    InvocationStatus,
    JsonObject,
)


class _RequestStore(InMemoryRuntimeStore):
    def __init__(self) -> None:
        super().__init__()
        self._sequence = 0

    def finish(self, invocation_id: str) -> None:
        """Terminalize on the owning event loop without a cancellation checkpoint."""
        state = self.states.get(invocation_id)
        if state is not None and state.status == InvocationStatus.ACTIVE:
            self.states[invocation_id] = replace(state, status=InvocationStatus.FAILED)
            self._append_event(invocation_id, state.attempt, {"type": "run.failed"})

    def _append_event(self, invocation_id: str, attempt: int, event: JsonObject) -> InvocationEvent:
        self._sequence += 1
        persisted = InvocationEvent(self._sequence, invocation_id, attempt, copy.deepcopy(event))
        self.persisted_events.append(persisted)
        return persisted


class _EventStream(StreamingResponse):
    def __init__(self, execution: RequestExecution, private_id: str, after: int) -> None:
        super().__init__(execution.events(private_id, after), media_type="text/event-stream")
        self.execution = execution
        self.private_id = private_id
        execution._pin(private_id)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.execution._unpin(self.private_id)


class _RequestStream(StreamingResponse):
    def __init__(
        self,
        execution: RequestExecution,
        private_id: str,
        invocation_id: str,
        payload: JsonObject,
        auth: RequestAuthContext,
        handler: InvocationHook,
        execute: bool,
    ) -> None:
        super().__init__(execution.events(private_id), media_type="text/event-stream")
        self.execution = execution
        self.private_id = private_id
        self.invocation_id = invocation_id
        self.payload = payload
        self.auth = auth
        self.handler = handler
        self.execute = execute

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        task = None
        try:
            if self.execute:
                task = asyncio.create_task(
                    self.execution._run(
                        self.private_id,
                        self.invocation_id,
                        self.payload,
                        self.auth,
                        self.handler,
                    )
                )
            await super().__call__(scope, receive, send)
        finally:
            self.execution._finish(self.private_id, self.auth)
            try:
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            finally:
                self.execution._unpin(self.private_id)


class RequestExecution:
    """Retain only token-free process-local outcomes; never enqueue execution."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 3600,
        retention_seconds: float = 3600,
        max_records: int = 256,
        max_events: int = 2048,
    ) -> None:
        self.store = _RequestStore()
        self.timeout_seconds = timeout_seconds
        self.retention_seconds = retention_seconds
        self.max_records = max_records
        self.max_events = max_events
        self._errors: dict[str, AuthError] = {}
        self._finished: dict[str, float] = {}
        self._streams: dict[str, int] = {}
        self._admission = asyncio.Lock()

    def _prune(self, *, reserve: bool = False) -> None:
        expired = {
            identifier
            for identifier, finished in self._finished.items()
            if identifier not in self._streams
            and time.monotonic() - finished >= self.retention_seconds
        }
        if reserve:
            for identifier in self._finished:
                if identifier in self._streams:
                    continue
                if len(self.store.states) - len(expired) < self.max_records:
                    break
                expired.add(identifier)
        for identifier in expired:
            self.store.states.pop(identifier, None)
            self._errors.pop(identifier, None)
            self._finished.pop(identifier, None)
        if expired:
            self.store.persisted_events[:] = [
                event for event in self.store.persisted_events if event.invocation_id not in expired
            ]

    def _pin(self, private_id: str) -> None:
        self._streams[private_id] = self._streams.get(private_id, 0) + 1

    def _unpin(self, private_id: str) -> None:
        self._streams[private_id] -= 1
        if not self._streams[private_id]:
            del self._streams[private_id]

    def _finish(self, private_id: str, auth: RequestAuthContext) -> None:
        auth.close()
        self.store.finish(private_id)
        self._finished.setdefault(private_id, time.monotonic())

    async def lookup(self, request: Request, invocation_id: str) -> tuple[str, Invocation]:
        auth = RequestAuthContext.from_headers(request.headers)
        try:
            private_id = auth.namespace("invocation", invocation_id)
        finally:
            auth.close()
        self._prune()
        state = await self.store.get(private_id)
        if state is None:
            raise HTTPException(404, "invocation not found")
        return private_id, replace(state, invocation_id=invocation_id)

    async def event_stream(
        self, request: Request, invocation_id: str, after: int = 0
    ) -> StreamingResponse:
        private_id, _ = await self.lookup(request, invocation_id)
        return _EventStream(self, private_id, after)

    async def invoke(
        self,
        request: Request,
        invocation_id: str,
        payload: JsonObject,
        handler: InvocationHook | None,
        *,
        background: bool,
        stream: bool,
    ) -> JSONResponse | StreamingResponse:
        auth = RequestAuthContext.from_headers(request.headers)
        transferred = False
        pinned_id = None
        try:
            if background:
                raise AuthError(
                    "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
                    "Request-user tools require synchronous or streaming execution",
                    400,
                )
            if handler is None:
                raise HTTPException(500, "no invocation handler is registered")
            private_id = auth.namespace("invocation", invocation_id)
            async with self._admission:
                self._prune(reserve=private_id not in self.store.states)
                if (
                    private_id not in self.store.states
                    and len(self.store.states) >= self.max_records
                ):
                    raise AuthError(
                        "MCP_USER_CAPACITY", "Too many active request-user invocations", 429
                    )
                state = await self.store.accept(private_id, payload)
                if state.status == InvocationStatus.ACTIVE:
                    raise InvocationConflictError("invocation is already active")
                claimed = await self.store.claim(private_id) if not state.is_terminal else None
                if not state.is_terminal and claimed is None:
                    raise InvocationConflictError("invocation is already active")
                self._pin(private_id)
                pinned_id = private_id
            if stream:
                response = _RequestStream(
                    self,
                    private_id,
                    invocation_id,
                    payload,
                    auth,
                    handler,
                    claimed is not None,
                )
                transferred = True
                return response
            if claimed is not None:
                task = asyncio.create_task(
                    self._run(private_id, invocation_id, payload, auth, handler)
                )
                watcher = asyncio.create_task(self._disconnect(request, task))
                try:
                    await task
                finally:
                    self._finish(private_id, auth)
                    watcher.cancel()
                    task.cancel()
                    await asyncio.gather(watcher, task, return_exceptions=True)
            return await self._response(private_id, invocation_id)
        finally:
            if not transferred:
                auth.close()
                if pinned_id is not None:
                    self._unpin(pinned_id)

    async def _disconnect(self, request: Request, task: asyncio.Task) -> None:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                return
            await asyncio.sleep(0.1)

    async def _run(
        self,
        private_id: str,
        invocation_id: str,
        payload: JsonObject,
        auth: RequestAuthContext,
        handler: InvocationHook,
    ) -> None:
        emitted = 0

        async def emit(event: JsonObject) -> int:
            nonlocal emitted
            if emitted >= self.max_events:
                raise AuthError("MCP_USER_EVENT_LIMIT", "Invocation event limit exceeded", 429)
            emitted += 1
            sequence = await self.store.append_event(private_id, 1, event)
            if sequence is None:
                raise RuntimeError("invocation is no longer active")
            return sequence

        context = InvocationContext(
            invocation_id=invocation_id,
            session_id=auth.namespace("session", str(payload["session_id"])),
            attempt=1,
            _attempt_context=InvocationAttemptContext(invocation_id, 1, emit),
            request_auth=auth,
        )
        try:
            output = await asyncio.wait_for(
                handler(copy.deepcopy(payload["input"]), context), self.timeout_seconds
            )
            await self.store.complete(private_id, 1, output)
        except asyncio.CancelledError:
            await self.store.fail(private_id, 1)
            raise
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, AuthError)
                else AuthError("MCP_USER_INVOCATION_FAILED", "Request-user invocation failed", 500)
            )
            self._errors[private_id] = AuthError(
                error.code, str(error), error.status_code, error.integration_id
            )
            await self.store.append_event(
                private_id, 1, {"type": "error", "error": dict(error.payload())}
            )
            await self.store.fail(private_id, 1)
        finally:
            self._finish(private_id, auth)

    async def _response(self, private_id: str, invocation_id: str) -> JSONResponse:
        state = await self.store.get(private_id)
        if state is not None and state.status == InvocationStatus.COMPLETED:
            return JSONResponse(
                {"id": invocation_id, "status": "completed", "output": state.response}
            )
        error = self._errors.get(
            private_id, AuthError("MCP_USER_INVOCATION_FAILED", "Invocation failed", 500)
        )
        return JSONResponse(
            {"id": invocation_id, "status": "failed", "error": error.payload()},
            status_code=error.status_code,
        )

    async def events(self, private_id: str, after: int = 0) -> AsyncIterator[str]:
        cursor = after
        while True:
            state = await self.store.get(private_id)
            for event in await self.store.events(private_id, cursor):
                cursor = event.sequence_number
                yield f"id: {cursor}\nevent: {event.event.get('type', 'message')}\ndata: {json.dumps(event.event)}\n\n"
            if state is None or state.is_terminal:
                return
            await asyncio.sleep(0.02)
