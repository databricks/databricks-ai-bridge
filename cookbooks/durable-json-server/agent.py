"""A complete agent using the SDK-provided generic durable server."""

import asyncio

from databricks_ai_bridge.durable_server import (
    DatabricksDurableServer,
    DurableRequestContext,
)


async def agent(payload: dict, context: DurableRequestContext) -> dict:
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


server = DatabricksDurableServer(agent)
app = server.app
