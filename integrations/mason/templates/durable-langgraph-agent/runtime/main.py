"""Expose the minimal LangGraph workload through Mason's durable application."""

import os

import uvicorn
from agent.agent import run_agent
from databricks_mason import DurableAgentApp

app = DurableAgentApp()


@app.invoke
async def invoke(input, context):
    return await run_agent(input, context)


@app.on_recovery
async def recover(input, context):
    recovery_input = {
        **input,
        "message": f"{input['message']} (recovery attempt after the pod crashed)",
    }
    return await run_agent(recovery_input, context)


def main() -> None:
    uvicorn.run(
        "runtime.main:app",
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
