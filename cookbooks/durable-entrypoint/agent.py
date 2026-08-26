"""OpenAI Agents SDK loop hosted by the SDK-provided durable entrypoint."""

import os

import uvicorn
from openai_agent import run_openai_agent

from databricks_ai_bridge.durable_app import DatabricksDurableApp, DurableAgentContext

app = DatabricksDurableApp()


@app.entrypoint
async def agent(payload: dict, context: DurableAgentContext) -> dict:
    output = await run_openai_agent(
        prompt=str(payload["prompt"]),
        session_id=context.session_id,
        emit=context.emit,
    )

    return {
        "output": output,
        "session_id": context.session_id,
        "attempt": context.attempt,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
