"""Expose the LangGraph workload through Mason's durable application."""

import os
from pathlib import Path

import uvicorn
from agent.agent import configure, run_agent
from databricks_mason import DurableAgentApp
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
configure()

app = DurableAgentApp()


@app.invoke
async def invoke(input, context):
    return await run_agent(input, context)


@app.on_recovery
async def recover(input, context):
    if not isinstance(input, list):
        raise ValueError("input must be a list of message objects")
    recovery_input = [
        {
            "role": "system",
            "content": (
                "This is a recovery attempt after the previous pod crashed. Continue the user's "
                "request, but do not repeat the wait_for_seconds tool used to demonstrate the crash."
            ),
        },
        *input,
    ]
    return await run_agent(recovery_input, context)


def main() -> None:
    uvicorn.run(
        "runtime.main:app",
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
