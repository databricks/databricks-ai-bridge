"""Durable MLflow AgentServer implementation."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlflow.genai.agent_server import AgentServer
from mlflow.genai.agent_server.server import RETURN_TRACE_HEADER
from mlflow.genai.agent_server.utils import set_request_headers
from mlflow.types.responses import ResponsesAgentRequest

from databricks_ai_bridge.durable_runtime import (
    DatabricksDurableRuntime,
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

_CURRENT_EXECUTION_CONTEXT: ContextVar[DurableExecutionContext | None] = ContextVar(
    "databricks_durable_execution_context",
    default=None,
)
_RETURN_TRACE_ID: ContextVar[bool] = ContextVar("databricks_durable_return_trace_id", default=False)
_RESPONSE_STATUS = {
    DurableExecutionStatus.QUEUED: "in_progress",
    DurableExecutionStatus.ACTIVE: "in_progress",
    DurableExecutionStatus.COMPLETED: "completed",
    DurableExecutionStatus.FAILED: "failed",
}


@dataclass(frozen=True)
class PreparedDurableRequest:
    """Validated AgentServer request adapted to durable execution."""

    execution_id: str
    payload: JsonObject


class DurableRequestPreparer(Protocol):
    def __call__(self, request: ResponsesAgentRequest) -> PreparedDurableRequest: ...


def get_durable_execution_context() -> DurableExecutionContext:
    """Return attempt metadata while the registered ``@invoke`` handler runs."""
    context = _CURRENT_EXECUTION_CONTEXT.get()
    if context is None:
        raise RuntimeError("durable execution context is only available inside @invoke")
    return context


class DatabricksDurableAgentServer(AgentServer):
    """MLflow AgentServer with durable blocking and background invocation."""

    _SUPPORTED_AGENT_TYPE: Literal["ResponsesAgent"] = "ResponsesAgent"

    def __init__(
        self,
        agent_type: Literal["ResponsesAgent"] = _SUPPORTED_AGENT_TYPE,
        *,
        prepare_request: DurableRequestPreparer,
        enable_chat_proxy: bool = False,
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
    ) -> None:
        if agent_type != self._SUPPORTED_AGENT_TYPE:
            raise ValueError(
                f"DatabricksDurableAgentServer only supports {self._SUPPORTED_AGENT_TYPE!r}, "
                f"got {agent_type!r}"
            )
        self.prepare_request = prepare_request
        self.on_startup = on_startup
        self.on_shutdown = on_shutdown
        self.runtime = DatabricksDurableRuntime(
            self._execute_durable,
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
        super().__init__(agent_type, enable_chat_proxy=enable_chat_proxy)
        self.app.router.lifespan_context = self._lifespan

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
        super()._setup_routes()
        durable_paths = {"/invocations", "/responses"}
        self.app.router.routes = [
            route
            for route in self.app.router.routes
            if not (
                getattr(route, "path", None) in durable_paths
                and "POST" in (getattr(route, "methods", None) or set())
            )
        ]
        self.app.add_api_route("/invocations", self._invoke_durable, methods=["POST"])
        self.app.add_api_route("/responses", self._invoke_durable, methods=["POST"])
        self.app.add_api_route("/responses/{execution_id}", self._retrieve_durable, methods=["GET"])
        self.app.add_api_route("/api/healthz", self._health, methods=["GET"])

    async def _invoke_durable(self, request: Request) -> JSONResponse:
        set_request_headers(dict(request.headers))
        data = await self._read_json_object(request)
        background = self._pop_boolean(data, "background")
        if self._pop_boolean(data, "stream"):
            raise HTTPException(400, "streaming is not supported")

        try:
            validated_request = self.validator.validate_and_convert_request(data)
            if not isinstance(validated_request, ResponsesAgentRequest):
                raise TypeError("ResponsesAgent validator returned an unexpected request type")
            prepared = self.prepare_request(validated_request)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

        if not prepared.execution_id:
            raise HTTPException(400, "prepared execution_id must not be empty")
        if not isinstance(prepared.payload, dict):
            raise HTTPException(400, "prepared payload must be a JSON object")

        trace_token = _RETURN_TRACE_ID.set(
            (request.headers.get(RETURN_TRACE_HEADER) or "").lower() == "true"
        )
        try:
            try:
                if background:
                    state = await self.runtime.submit(prepared.execution_id, prepared.payload)
                    status_code = 200 if state.is_terminal else 202
                    return JSONResponse(self._status_response(state), status_code=status_code)
                response = await self.runtime.invoke(prepared.execution_id, prepared.payload)
                return JSONResponse(response)
            except DurableRequestConflictError as exc:
                raise HTTPException(409, str(exc)) from exc
            except DurableExecutionFailedError as exc:
                raise HTTPException(500, str(exc)) from exc
        finally:
            _RETURN_TRACE_ID.reset(trace_token)

    async def _retrieve_durable(self, execution_id: str) -> JSONResponse:
        state = await self.runtime.get(execution_id)
        if state is None:
            raise HTTPException(404, f"execution {execution_id!r} was not found")
        return JSONResponse(self._status_response(state))

    async def _execute_durable(
        self,
        request: JsonObject,
        context: DurableExecutionContext,
    ) -> JsonObject:
        validated_request = self.validator.validate_and_convert_request(copy.deepcopy(request))
        if not isinstance(validated_request, ResponsesAgentRequest):
            raise TypeError("ResponsesAgent validator returned an unexpected request type")
        token = _CURRENT_EXECUTION_CONTEXT.set(context)
        try:
            response = await super()._handle_invoke_request(
                validated_request,
                _RETURN_TRACE_ID.get(),
            )
        finally:
            _CURRENT_EXECUTION_CONTEXT.reset(token)
        return self._json_object(response)

    @staticmethod
    def _status_response(state: DurableExecution) -> JsonObject:
        if state.status == DurableExecutionStatus.COMPLETED:
            if state.response is None:
                raise RuntimeError(f"execution {state.execution_id!r} completed without a response")
            return copy.deepcopy(state.response)
        custom_inputs = dict(state.request.get("custom_inputs") or {})
        custom_outputs: JsonObject = {
            "execution_id": state.execution_id,
            "attempt": state.attempt,
        }
        if custom_inputs.get("session_id"):
            custom_outputs["session_id"] = custom_inputs["session_id"]
        return {
            "id": state.execution_id,
            "status": _RESPONSE_STATUS[state.status],
            "output": [],
            "custom_outputs": custom_outputs,
        }

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

    @staticmethod
    def _json_object(value: Any) -> JsonObject:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", exclude_none=True)
        if not isinstance(value, dict):
            raise TypeError(f"expected a JSON object, got {type(value).__name__}")
        return value
