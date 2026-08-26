"""A complete durable agent using the SDK-provided entrypoint application."""

import asyncio
import os

import uvicorn

from databricks_ai_bridge.durable_app import DatabricksDurableApp, DurableAgentContext

app = DatabricksDurableApp()


@app.entrypoint
async def agent(payload: dict, context: DurableAgentContext) -> dict:
    steps = int(payload.get("steps", 3))
    delay_seconds = float(payload.get("delay_seconds", 1))

    for step in range(1, steps + 1):
        await asyncio.sleep(delay_seconds)
        await context.emit({"type": "progress", "step": step, "total": steps})

    return {
        "message": "completed",
        "session_id": context.session_id,
        "attempt": context.attempt,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
