"""LangGraph agent using durable event-log recovery."""

import os

import langgraph_agent_shared.handlers  # noqa: F401
import uvicorn
from langgraph_agent_shared.runtime import create_app

from databricks_ai_bridge.long_running import ResumeStrategy

app = create_app(ResumeStrategy.EVENT_LOG)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
