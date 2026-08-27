"""Browser UI and demo controls for a Mason agent project."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_UI_ROOT = Path(__file__).resolve().parent.parent / "ui"
_INSTANCE_ID = uuid.uuid4().hex[:12]
_CRASH_ENV = "MASON_DEMO_CRASH_ENABLED"


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _execution_identity() -> str:
    if os.getenv("DATABRICKS_APP_NAME"):
        return "Databricks App service principal"
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    return (
        f"Local Databricks profile: {profile}" if profile else "Databricks default authentication"
    )


def install_ui(app: FastAPI) -> None:
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
        return {
            "instance_id": _INSTANCE_ID,
            "viewer": viewer,
            "execution_identity": _execution_identity(),
            "streaming": {"enabled": True, "transport": "Server-sent events"},
            "background": {"enabled": True, "durable": False},
            "session": {
                "durable": bool(os.getenv("AGENT_SESSION_STORE")),
                "mode": "Managed Session Store"
                if os.getenv("AGENT_SESSION_STORE")
                else "In-process checkpointer",
            },
            "memory": {
                "enabled": bool(os.getenv("AGENT_MEMORY_STORE")),
                "actor": os.getenv("AGENT_MEMORY_ACTOR_ID", "agent"),
            },
            "crash": {
                "enabled": _enabled(_CRASH_ENV),
                "restart_managed": bool(os.getenv("DATABRICKS_APP_NAME")),
            },
        }

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
