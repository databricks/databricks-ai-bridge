"""Shared LongRunningAgentServer lifecycle wiring."""

import os
from contextlib import asynccontextmanager

import uvicorn

from databricks_ai_bridge.long_running import LongRunningAgentServer

import handlers  # noqa: F401
from sessions import initialize_sessions


def create_server(*, auto_recovery: bool, sse_replay: bool) -> LongRunningAgentServer:
    server = LongRunningAgentServer(
        "ResponsesAgent",
        db_autoscaling_endpoint=os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT"),
        task_timeout_seconds=3600,
        heartbeat_interval_seconds=3,
        heartbeat_stale_threshold_seconds=10,
        auto_recovery=auto_recovery,
        sse_replay=sse_replay,
    )
    original_lifespan = server.app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        await initialize_sessions()
        async with original_lifespan(application):
            yield

    server.app.router.lifespan_context = lifespan

    @server.app.get("/api/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "healthy"}

    return server


def run_server(app_import_string: str) -> None:
    uvicorn.run(
        app_import_string,
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
