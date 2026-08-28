"""Browser UI and managed-state demo controls for a Mason agent project."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.mason import recovery
from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_UI_ROOT = Path(__file__).resolve().parent.parent / "ui"
_INSTANCE_ID = recovery.process_id()
_CRASH_ENV = "MASON_DEMO_CRASH_ENABLED"
_MEMORY_STORE_ENV = "AGENT_MEMORY_STORE"
_MEMORY_ACTOR_ENV = "AGENT_MEMORY_ACTOR_ID"
_SESSION_STORE_ENV = "AGENT_SESSION_STORE"
_SESSION_ACTOR_ENV = "AGENT_SESSION_ACTOR_ID"
_PROFILE_CONFLICT_ENV = ("DATABRICKS_CONFIG_PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN")
_AGENTS_API = "/api/agents/v1"

SessionHistoryHandler = Callable[[str], Awaitable[dict[str, Any]]]


class MemoryEntryRequest(BaseModel):
    path: str = Field(min_length=1, pattern=r"^/")
    content: str = Field(min_length=1)
    description: str | None = None
    session_id: str | None = None


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=1)


class SessionItemsRequest(BaseModel):
    items: list[dict[str, Any]] = Field(min_length=1)


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _execution_identity() -> str:
    if os.getenv("DATABRICKS_APP_NAME"):
        return "Databricks App service principal"
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    return (
        f"Local Databricks profile: {profile}" if profile else "Databricks default authentication"
    )


def _memory_store() -> str:
    store = os.getenv(_MEMORY_STORE_ENV, "").strip().strip("/")
    return store.removeprefix("memory-stores/")


def _memory_actor() -> str:
    return os.getenv(_MEMORY_ACTOR_ENV, "agent")


def _session_store() -> str:
    return os.getenv(_SESSION_STORE_ENV, "").strip()


def _session_actor() -> str:
    return os.getenv(_SESSION_ACTOR_ENV) or _memory_actor()


class _ManagedStateClient:
    def __init__(self) -> None:
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
        if profile:
            inherited = {name: os.environ.pop(name, None) for name in _PROFILE_CONFLICT_ENV}
            try:
                self._workspace = WorkspaceClient(profile=profile)
            finally:
                for name, value in inherited.items():
                    if value is not None:
                        os.environ[name] = value
        else:
            self._workspace = WorkspaceClient()

    def _do(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict:
        headers = None
        if self._workspace.config.workspace_id:
            headers = {"X-Databricks-Workspace-Id": str(self._workspace.config.workspace_id)}
        result = self._workspace.api_client.do(
            method, path, query=query, body=body, headers=headers
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                f"Expected an object response from {path}, got {type(result).__name__}"
            )
        return result

    def create_memory_entry(self, request: MemoryEntryRequest) -> dict:
        body = {
            "actor_id": _memory_actor(),
            "path": request.path,
            "content": request.content,
        }
        if request.description:
            body["description"] = request.description
        if request.session_id:
            body["session_id"] = request.session_id
        return self._do("POST", f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries", body=body)

    def list_memory_entries(self, path_prefix: str | None = None) -> dict:
        query = {"actor_id": _memory_actor(), "page_size": 100}
        if path_prefix:
            query["path_prefix"] = path_prefix
        return self._do(
            "GET", f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries", query=query
        )

    def search_memory_entries(self, request: MemorySearchRequest) -> dict:
        return self._do(
            "POST",
            f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries:search",
            body={
                "actor_id": _memory_actor(),
                "query": request.query,
                "limit": request.limit,
            },
        )

    def ensure_session(self, session_id: str) -> dict:
        try:
            return self._do(
                "POST",
                f"{_AGENTS_API}/session-stores/{_session_store()}/sessions",
                query={"session_id": session_id},
                body={"actor_id": _session_actor(), "metadata": {"client": "mason-demo-ui"}},
            )
        except Exception as exc:
            code = str(getattr(exc, "error_code", "")).upper()
            already_exists = code in {"ALREADY_EXISTS", "RESOURCE_ALREADY_EXISTS"}
            if not already_exists and "already exists" not in str(exc).lower():
                raise
            return self._do(
                "GET",
                f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}",
            )

    def get_session(self, session_id: str) -> dict:
        return self._do(
            "GET", f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}"
        )

    def append_session_items(self, session_id: str, items: list[dict[str, Any]]) -> dict:
        return self._do(
            "POST",
            f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}/items:append",
            body={"items": [{"data": item} for item in items]},
        )

    def list_session_items(self, session_id: str) -> dict:
        return self._do(
            "GET",
            f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}/items",
            query={"order_by": "create_time asc", "page_size": 100},
        )


@lru_cache(maxsize=1)
def _state_client() -> _ManagedStateClient:
    return _ManagedStateClient()


async def _managed_call(operation, *args):
    try:
        return await asyncio.to_thread(operation, *args)
    except HTTPException:
        raise
    except Exception as exc:
        code = getattr(exc, "error_code", None)
        detail = f"{code}: {exc}" if code else str(exc)
        raise HTTPException(status_code=502, detail=detail) from exc


def _require_memory() -> None:
    if not _memory_store():
        raise HTTPException(
            status_code=503,
            detail=f"Set {_MEMORY_STORE_ENV} by deploying with --with-memory-store.",
        )


def _require_session() -> None:
    if not _session_store():
        raise HTTPException(
            status_code=503,
            detail=f"Set {_SESSION_STORE_ENV} by deploying with --with-session-store.",
        )


def _require_recovery() -> None:
    if not _enabled(_CRASH_ENV):
        raise HTTPException(
            status_code=503,
            detail=f"Run `mason add ui --enable-crash` or set {_CRASH_ENV}=true.",
        )
    _require_session()


async def _recovery_call(handler, session_id: str) -> dict[str, Any]:
    try:
        return await handler(session_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def install_ui(app: FastAPI, session_history: SessionHistoryHandler | None = None) -> None:
    """Mount the Mason demo UI and its runtime control endpoints."""
    app.mount("/ui-assets", StaticFiles(directory=_UI_ROOT), name="mason-demo-ui-assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_UI_ROOT / "index.html")

    @app.get("/api/demo/config", include_in_schema=False)
    async def demo_config(request: Request) -> dict:
        viewer = (
            request.headers.get("x-forwarded-email")
            or request.headers.get("x-forwarded-user")
            or "Local developer"
        )
        memory_store = _memory_store()
        session_store = _session_store()
        return {
            "instance_id": _INSTANCE_ID,
            "viewer": viewer,
            "execution_identity": _execution_identity(),
            "streaming": {"enabled": True, "transport": "Server-sent events"},
            "background": {"enabled": True, "durable": False},
            "session": {
                "durable": bool(session_store),
                "managed": bool(session_store),
                "history": bool(session_store or session_history),
                "mode": "Managed Session Store" if session_store else "In-process checkpointer",
                "store": session_store or None,
                "actor": _session_actor(),
            },
            "memory": {
                "enabled": bool(memory_store),
                "store": f"memory-stores/{memory_store}" if memory_store else None,
                "actor": _memory_actor(),
            },
            "crash": {
                "enabled": _enabled(_CRASH_ENV),
                "restart_managed": bool(os.getenv("DATABRICKS_APP_NAME")),
            },
            "recovery": {
                "enabled": bool(session_store and _enabled(_CRASH_ENV)),
                "automatic_resume": True,
                "steps": recovery.step_names(),
                "step_seconds": recovery.step_seconds(),
            },
        }

    @app.post("/api/demo/memory/entries", include_in_schema=False)
    async def create_memory_entry(request: MemoryEntryRequest) -> dict:
        _require_memory()
        return await _managed_call(_state_client().create_memory_entry, request)

    @app.get("/api/demo/memory/entries", include_in_schema=False)
    async def list_memory_entries(path_prefix: str | None = Query(default=None)) -> dict:
        _require_memory()
        return await _managed_call(_state_client().list_memory_entries, path_prefix)

    @app.post("/api/demo/memory/search", include_in_schema=False)
    async def search_memory_entries(request: MemorySearchRequest) -> dict:
        _require_memory()
        return await _managed_call(_state_client().search_memory_entries, request)

    @app.post("/api/demo/sessions", include_in_schema=False)
    async def ensure_session(request: SessionRequest) -> dict:
        _require_session()
        return await _managed_call(_state_client().ensure_session, request.session_id)

    @app.get("/api/demo/sessions/{session_id}", include_in_schema=False)
    async def get_session(session_id: str) -> dict:
        _require_session()
        return await _managed_call(_state_client().get_session, session_id)

    @app.post("/api/demo/sessions/{session_id}/items", include_in_schema=False)
    async def append_session_items(session_id: str, request: SessionItemsRequest) -> dict:
        _require_session()
        return await _managed_call(_state_client().append_session_items, session_id, request.items)

    @app.get("/api/demo/sessions/{session_id}/items", include_in_schema=False)
    async def list_session_items(session_id: str) -> dict:
        if _session_store():
            return await _managed_call(_state_client().list_session_items, session_id)
        if session_history is None:
            raise HTTPException(status_code=503, detail="Session history is not available.")
        return await session_history(session_id)

    @app.get("/api/demo/recovery/{session_id}", include_in_schema=False)
    async def recovery_status(session_id: str) -> dict:
        _require_recovery()
        return await _recovery_call(recovery.status, session_id)

    @app.post("/api/demo/recovery/{session_id}/start", include_in_schema=False)
    async def start_recovery(session_id: str) -> dict:
        _require_recovery()
        return await _recovery_call(recovery.start, session_id)

    @app.post("/api/demo/recovery/{session_id}/resume", include_in_schema=False)
    async def resume_recovery(session_id: str) -> dict:
        _require_recovery()
        return await _recovery_call(recovery.resume, session_id)

    @app.post("/api/demo/crash", include_in_schema=False)
    async def crash() -> dict:
        if not _enabled(_CRASH_ENV):
            raise HTTPException(
                status_code=403,
                detail=f"Set {_CRASH_ENV}=true to enable the demo crash endpoint.",
            )
        asyncio.get_running_loop().call_later(0.5, os._exit, 86)
        return {
            "status": "crashing",
            "instance_id": _INSTANCE_ID,
            "restart_managed": bool(os.getenv("DATABRICKS_APP_NAME")),
        }
