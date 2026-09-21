"""Browser UI and managed-state demo controls for a Mason agent project."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from databricks_mason import workspace_client
from databricks_mason.runtime.model_services import list_ai_gateway_model_services
from databricks_mason.runtime.store import runtime_store_is_persistent_environment

_UI_ROOT = Path(__file__).resolve().parent.parent / "ui"
_INSTANCE_ID = uuid.uuid4().hex[:12]  # identifies this process in the UI
_AGENTS_API = "/api/2.0/agents"
_MESSAGE_ROLES = {
    "ai",
    "assistant",
    "developer",
    "function",
    "human",
    "human_decision",
    "system",
    "tool",
    "user",
}


class MemoryEntryRequest(BaseModel):
    path: str = Field(min_length=1, pattern=r"^/")
    content: str = Field(min_length=1)
    description: str | None = None


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    actor: str | None = None


class SessionItemsRequest(BaseModel):
    items: list[dict[str, Any]] = Field(min_length=1)


_USER_HEADERS = ("x-forwarded-email", "x-forwarded-user")


def _memory_store() -> str:
    # Same resolution the agent uses (AGENT_MEMORY_STORE env → agent.toml binding), so the demo
    # panels reflect exactly the store the agent reads/writes.
    from databricks_mason.runtime.tool_manifest import resolve_memory_store

    return (resolve_memory_store() or "").strip().strip("/")


def _session_store() -> str:
    from databricks_mason.runtime.tool_manifest import resolve_session_store

    return (resolve_session_store() or "").strip()


def _request_actor(request: Request) -> str:
    """The actor for a demo request — the signed-in user, so the panels show that user's own data.

    Mirrors how the agent resolves its actor (same forwarded-identity headers), so the memory and
    session views here list exactly what the agent reads/writes for the current user. Falls back to
    ``"agent"`` locally / when unauthenticated.
    """
    for header in _USER_HEADERS:
        if value := request.headers.get(header):
            return value
    return "agent"


def _request_session_id(request: Request) -> str:
    """Read the chat session selected by the browser, falling back to the router cookie locally."""
    session_id = request.query_params.get("session_id") or getattr(
        request.state, "session_id", None
    )
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return str(session_id)


def _is_deployed() -> bool:
    app_url = os.getenv("DATABRICKS_APP_URL", "")
    is_local = app_url.startswith(("http://localhost", "http://127.0.0.1"))
    return bool(os.getenv("DATABRICKS_APP_NAME")) and not is_local


# Tracing turns on only with both a destination and an experiment, mirroring
# databricks_mason.runtime.tracing so the UI card matches what the agent actually does.
_TRACING_DESTINATION_VARS = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACING_DESTINATION")


def _workspace_host() -> str:
    """Best-effort workspace host for building MLflow links; never raises."""
    try:
        host = workspace_client().config.host or ""
    except Exception:  # noqa: BLE001 - host is best-effort; a failure must not break the config route
        host = os.getenv("DATABRICKS_HOST", "")
    return host.rstrip("/")


def _tracing() -> dict:
    """Whether MLflow tracing is configured, plus a best-effort link to its experiment."""
    destination = ""
    for var in _TRACING_DESTINATION_VARS:
        if value := os.getenv(var):
            destination = value
            break
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip()
    enabled = bool(destination) and bool(experiment_id or experiment_name)
    url: str | None = None
    if enabled:
        if destination.startswith(("http://", "https://")) and "databricks" not in destination:
            # A local OSS MLflow server: link into its own UI.
            base = destination.rstrip("/")
            url = f"{base}/#/experiments/{experiment_id}" if experiment_id else base
        else:
            host = _workspace_host()
            if host:
                url = (
                    f"{host}/ml/experiments/{experiment_id}"
                    if experiment_id
                    else f"{host}/ml/experiments"
                )
    return {"enabled": enabled, "experiment_id": experiment_id or None, "url": url}


# Cap the picker: the gateway's schema can list more models than belong in a dropdown.
_MODEL_LIMIT = 20


def _default_model() -> str:
    """The agent's configured default endpoint (``agent.agent.MODEL``), imported lazily.

    Imported inside the function, not at module load, so the light UI import path doesn't pull in the
    agent stack (matching ``_local_history`` below).
    """
    from agent.agent import MODEL

    return MODEL


def _rank_models(default: str, names: list[str]) -> list[str]:
    """Pin the default first, then the remaining gateway models alphabetically, without duplicates.

    Truncation therefore only ever sheds models beyond the cap, never the configured default.
    """
    ordered = [default, *sorted(n for n in names if n != default)]
    seen: set[str] = set()
    return [n for n in ordered if not (n in seen or seen.add(n))]


def _discover_chat_models() -> list[str]:
    """The AI Gateway chat models for the picker: default first, then the rest, capped.

    Best-effort: if listing fails (missing permission, repeated transient errors), fall back to just
    the default so the picker still works. The default is always present and first.
    """
    default = _default_model()
    try:
        names = list_ai_gateway_model_services(workspace_client())
    except Exception:  # noqa: BLE001 - a broken listing must not break the whole config endpoint
        names = []
    return _rank_models(default, names)[:_MODEL_LIMIT]


class _ManagedStateClient:
    def __init__(self) -> None:
        self._workspace = workspace_client()

    def _do(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict:
        result = self._workspace.api_client.do(method, path, query=query, body=body)
        if not isinstance(result, dict):
            raise RuntimeError(
                f"Expected an object response from {path}, got {type(result).__name__}"
            )
        return result

    def create_memory_entry(self, actor: str, request: MemoryEntryRequest, session_id: str) -> dict:
        body = {
            "actor_id": actor,
            "path": request.path,
            "content": request.content,
            "session_id": session_id,
        }
        if request.description:
            body["description"] = request.description
        return self._do("POST", f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries", body=body)

    def list_memory_entries(self, actor: str, path_prefix: str | None = None) -> dict:
        query = {"actor_id": actor, "page_size": 100}
        if path_prefix:
            query["path_prefix"] = path_prefix
        return self._do(
            "GET", f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries", query=query
        )

    def search_memory_entries(self, actor: str, request: MemorySearchRequest) -> dict:
        return self._do(
            "POST",
            f"{_AGENTS_API}/memory-stores/{_memory_store()}/entries:search",
            body={
                "actor_id": actor,
                "query": request.query,
                "limit": request.limit,
            },
        )

    def ensure_session(self, actor: str, session_id: str) -> dict:
        try:
            return self._do(
                "POST",
                f"{_AGENTS_API}/session-stores/{_session_store()}/sessions",
                query={"session_id": session_id},
                body={
                    "actor_id": actor,
                    "metadata": {"client": "mason-demo-ui"},
                },
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
            "GET",
            f"{_AGENTS_API}/session-stores/{_session_store()}/sessions/{session_id}",
        )

    def list_sessions(self, actor: str) -> dict:
        return self._do(
            "GET",
            f"{_AGENTS_API}/session-stores/{_session_store()}/sessions",
            query={
                "filter": f"actor_id = {json.dumps(actor)}",
                "order_by": "last_activity_time desc",
                "page_size": 50,
            },
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
            detail="No memory store configured. Run `mason memory bind <store>`.",
        )


def _require_session() -> None:
    if not _session_store():
        raise HTTPException(
            status_code=503,
            detail="No session store configured. Run `mason sessions bind <store>`.",
        )


async def _local_history(session_id: str) -> dict[str, Any]:
    """Reconstruct the transcript from the in-process session (no managed Session Store).

    Reads the Responses items the agent stored in its ``SQLiteSession`` for this browser session and
    shapes each into a ``{item_id, data}`` entry the UI renders. There are no durable interrupts here:
    a paused human-in-the-loop run is held in-process by ``agent.py`` and is not part of the session
    transcript, so ``interrupts`` is always empty for the unmanaged path.
    """
    from databricks_mason.openai.sessions import session_store

    session = session_store(session_id)
    items = []
    for index, message in enumerate(await session.get_items()):
        data = message if isinstance(message, dict) else {"content": str(message)}
        items.append({"item_id": str(data.get("id") or index), "data": data})
    return {"session_id": session_id, "session_items": items, "interrupts": []}


def _chat_sessions(result: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = []
    for session in result.get("sessions", []):
        if not isinstance(session, dict):
            continue
        metadata = session.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("public_session_id"):
            continue
        sessions.append(session)
    return sessions


def _chat_session_items(result: dict[str, Any]) -> dict[str, Any]:
    items = []
    for item in result.get("session_items", []):
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, dict) or data.get("event_type") or "content" not in data:
            continue
        role = str(data.get("role") or data.get("type") or "").lower()
        if role in _MESSAGE_ROLES:
            items.append(item)
    return {**result, "session_items": items}


def install_ui(app: FastAPI) -> None:
    """Mount the Mason UI and its runtime control endpoints."""
    app.mount("/ui-assets", StaticFiles(directory=_UI_ROOT), name="mason-demo-ui-assets")

    @app.middleware("http")
    async def disable_ui_caching(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/ui-assets/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_UI_ROOT / "index.html")

    @app.get("/api/ui/config", include_in_schema=False)
    async def ui_config(request: Request) -> dict:
        actor = _request_actor(request)
        memory_store = _memory_store()
        session_store = _session_store()
        default_model = _default_model()
        runtime_store_persistent = runtime_store_is_persistent_environment()
        runtime_store_mode = (
            "Runtime Store" if runtime_store_persistent else "In-process Runtime Store"
        )
        return {
            "session_id": _request_session_id(request),
            "instance_id": _INSTANCE_ID,
            "viewer": actor if actor != "agent" else "Local developer",
            "deployed": _is_deployed(),
            "models": {"default": default_model, "available": [default_model]},
            "streaming": {
                "enabled": True,
                "transport": "Server-sent events",
                "persistent": runtime_store_persistent,
                "mode": runtime_store_mode,
            },
            "background": {
                "enabled": True,
                "persistent": runtime_store_persistent,
                "mode": runtime_store_mode,
            },
            "session": {
                "durable": bool(session_store),
                "managed": bool(session_store),
                "history": True,
                "mode": "Managed Session Store" if session_store else "In-process session",
                "store": session_store or None,
                "actor": actor,
            },
            "memory": {
                "enabled": bool(memory_store),
                "store": f"memory-stores/{memory_store}" if memory_store else None,
                "actor": actor,
            },
            "tracing": _tracing(),
        }

    @app.get("/api/demo/models", include_in_schema=False)
    async def demo_models() -> dict:
        # Model discovery can take several seconds in a large workspace. Keep it separate from the
        # runtime config so the rest of the UI becomes interactive immediately.
        available_models = await asyncio.to_thread(_discover_chat_models)
        return {"default": _default_model(), "available": available_models}

    @app.post("/api/demo/memory/entries", include_in_schema=False)
    async def create_memory_entry(request: Request, payload: MemoryEntryRequest) -> dict:
        _require_memory()
        return await _managed_call(
            _state_client().create_memory_entry,
            _request_actor(request),
            payload,
            _request_session_id(request),
        )

    @app.get("/api/demo/memory/entries", include_in_schema=False)
    async def list_memory_entries(
        request: Request,
        path_prefix: str | None = Query(default=None),
        actor: str | None = Query(default=None),
    ) -> dict:
        # The UI can browse another actor's memories by passing ?actor=; default to the viewer.
        _require_memory()
        return await _managed_call(
            _state_client().list_memory_entries, actor or _request_actor(request), path_prefix
        )

    @app.post("/api/demo/memory/search", include_in_schema=False)
    async def search_memory_entries(request: Request, payload: MemorySearchRequest) -> dict:
        # payload.actor lets the UI search another actor's memories; default to the viewer.
        _require_memory()
        return await _managed_call(
            _state_client().search_memory_entries, payload.actor or _request_actor(request), payload
        )

    @app.post("/api/demo/sessions", include_in_schema=False)
    async def ensure_session(request: Request) -> dict:
        _require_session()
        return await _managed_call(
            _state_client().ensure_session,
            _request_actor(request),
            _request_session_id(request),
        )

    @app.get("/api/demo/sessions", include_in_schema=False)
    async def list_sessions(request: Request) -> dict:
        session_id = _request_session_id(request)
        actor = _request_actor(request)
        if not _session_store():
            return {
                "sessions": [
                    {
                        "session_id": session_id,
                        "actor_id": actor,
                        "metadata": {"client": "mason-demo-ui-local"},
                    }
                ],
                "current_session_id": session_id,
                "managed": False,
            }
        result = await _managed_call(_state_client().list_sessions, actor)
        return {
            **result,
            "sessions": _chat_sessions(result),
            "current_session_id": session_id,
            "managed": True,
        }

    @app.post("/api/demo/sessions/{session_id}/open", include_in_schema=False)
    async def open_session(request: Request, session_id: str) -> JSONResponse:
        _require_session()
        session = await _managed_call(_state_client().get_session, session_id)
        if session.get("actor_id") != _request_actor(request):
            raise HTTPException(status_code=403, detail="Session belongs to another actor.")
        return JSONResponse(
            {
                "session_id": session_id,
                "previous_session_id": _request_session_id(request),
                "managed": True,
            }
        )

    @app.get("/api/demo/session", include_in_schema=False)
    async def get_session(request: Request) -> dict:
        _require_session()
        return await _managed_call(_state_client().get_session, _request_session_id(request))

    @app.post("/api/demo/session/items", include_in_schema=False)
    async def append_session_items(request: Request, payload: SessionItemsRequest) -> dict:
        _require_session()
        return await _managed_call(
            _state_client().append_session_items,
            _request_session_id(request),
            payload.items,
        )

    @app.get("/api/demo/session/items", include_in_schema=False)
    async def list_session_items(request: Request) -> dict:
        session_id = _request_session_id(request)
        if _session_store():
            result = await _managed_call(_state_client().list_session_items, session_id)
            return _chat_session_items(result)
        return await _local_history(session_id)
