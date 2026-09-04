"""Expose the minimal LangGraph workload through Mason's durable application."""

import os

import uvicorn
from agent.agent import run_agent
from databricks_mason import DurableAgentApp

server = DurableAgentApp(run_agent, on_resume=run_agent)
app = server.app


def main() -> None:
    uvicorn.run(
        "runtime.main:app",
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
