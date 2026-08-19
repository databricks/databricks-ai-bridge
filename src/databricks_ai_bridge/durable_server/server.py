"""FastAPI server for durable JSON request execution."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from databricks_ai_bridge.durable_runtime import (
    DatabricksDurableRuntime,
    DurabilityStore,
    DurableExecution,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableExecutor,
    DurableRequestConflictError,
    JsonObject,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


@dataclass(frozen=True)
class PreparedDurableRequest:
    """Transport request adapted to the durable runtime contract."""

    execution_id: str
    payload: JsonObject


class DurableRequestPreparer(Protocol):
    def __call__(self, request: JsonObject) -> PreparedDurableRequest: ...


class DurableStatusResponse(Protocol):
    def __call__(self, state: DurableExecution) -> JsonObject: ...


def _default_status_response(state: DurableExecution) -> JsonObject:
    if state.status == DurableExecutionStatus.COMPLETED:
        if state.response is None:
            raise RuntimeError(f"execution {state.execution_id!r} completed without a response")
        return copy.deepcopy(state.response)
    return {
        "execution_id": state.execution_id,
        "status": state.status.value,
        "attempt": state.attempt,
    }


class DatabricksDurableServer:
    """Serve a :class:`DatabricksDurableRuntime` over HTTP.

    The server owns FastAPI lifecycle and the blocking, background, and
    retrieval routes. Callers adapt transport JSON to an execution ID and
    persisted payload, and may customize non-completed status responses.
    """

    def __init__(
        self,
        executor: DurableExecutor,
        *,
        prepare_request: DurableRequestPreparer,
        status_response: DurableStatusResponse = _default_status_response,
        on_startup: Callable[[], Awaitable[None]] | None = None,
        on_shutdown: Callable[[], Awaitable[None]] | None = None,
        durability_store: DurabilityStore | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = "databricks_durable_runtime",
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
        title: str = "Databricks Durable Server",
    ) -> None:
        self.prepare_request = prepare_request
        self.status_response = status_response
        self.on_startup = on_startup
        self.on_shutdown = on_shutdown
        self.runtime = DatabricksDurableRuntime(
            executor,
            durability_store=durability_store,
            autoscaling_endpoint=autoscaling_endpoint,
            project=project,
            branch=branch,
            workspace_client=workspace_client,
            schema=schema,
            heartbeat_seconds=heartbeat_seconds,
            stale_seconds=stale_seconds,
            scan_seconds=scan_seconds,
            poll_seconds=poll_seconds,
        )
        self.app = FastAPI(title=title, lifespan=self._lifespan)
        self._setup_routes()

    @asynccontextmanager
    async def _lifespan(self, application: FastAPI):
        if self.on_startup is not None:
            await self.on_startup()
        await self.runtime.start()
        try:
            yield
        finally:
            await self.runtime.stop()
            if self.on_shutdown is not None:
                await self.on_shutdown()

    def _setup_routes(self) -> None:
        self.app.add_api_route("/invocations", self._invoke, methods=["POST"])
        self.app.add_api_route("/responses", self._invoke, methods=["POST"])
        self.app.add_api_route("/responses/{execution_id}", self._retrieve, methods=["GET"])
        self.app.add_api_route("/health", self._health, methods=["GET"])
        self.app.add_api_route("/api/healthz", self._health, methods=["GET"])

    async def _invoke(self, request: Request) -> JSONResponse:
        data = await self._read_json_object(request)
        background = self._pop_boolean(data, "background")
        if self._pop_boolean(data, "stream"):
            raise HTTPException(400, "streaming is not supported")

        try:
            prepared = self.prepare_request(data)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

        if not prepared.execution_id:
            raise HTTPException(400, "prepared execution_id must not be empty")
        if not isinstance(prepared.payload, dict):
            raise HTTPException(400, "prepared payload must be a JSON object")

        try:
            if background:
                state = await self.runtime.submit(prepared.execution_id, prepared.payload)
                status_code = 200 if state.is_terminal else 202
                return JSONResponse(self.status_response(state), status_code=status_code)
            response = await self.runtime.invoke(prepared.execution_id, prepared.payload)
            return JSONResponse(response)
        except DurableRequestConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except DurableExecutionFailedError as exc:
            raise HTTPException(500, str(exc)) from exc

    async def _retrieve(self, execution_id: str) -> JSONResponse:
        state = await self.runtime.get(execution_id)
        if state is None:
            raise HTTPException(404, f"execution {execution_id!r} was not found")
        return JSONResponse(self.status_response(state))

    async def _health(self) -> JsonObject:
        return {"status": "healthy"}

    @staticmethod
    async def _read_json_object(request: Request) -> JsonObject:
        try:
            data = await request.json()
        except Exception as exc:
            raise HTTPException(400, f"invalid JSON request body: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(400, "request body must be a JSON object")
        return data

    @staticmethod
    def _pop_boolean(data: JsonObject, key: str) -> bool:
        value = data.pop(key, False)
        if not isinstance(value, bool):
            raise HTTPException(400, f"{key} must be a boolean")
        return value
