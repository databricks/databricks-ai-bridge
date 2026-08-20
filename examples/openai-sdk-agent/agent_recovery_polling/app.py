"""Agent-session-managed recovery with polling and no event log."""

import os
from contextlib import asynccontextmanager

import handlers  # noqa: F401
import uvicorn
from openai_sdk_agent_shared.sessions import initialize_sessions

from databricks_ai_bridge.long_running import LongRunningAgentServer

agent_server = LongRunningAgentServer(
    "ResponsesAgent",
    db_autoscaling_endpoint=os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT"),
    task_timeout_seconds=3600,
    heartbeat_interval_seconds=3,
    heartbeat_stale_threshold_seconds=10,
    auto_recovery=False,
    sse_replay=False,
)
original_lifespan = agent_server.app.router.lifespan_context


@asynccontextmanager
async def lifespan(application):
    await initialize_sessions()
    async with original_lifespan(application):
        yield


agent_server.app.router.lifespan_context = lifespan


@agent_server.app.get("/api/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "healthy"}


app = agent_server.app


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
