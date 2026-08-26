"""A tiny agent used to compare durable execution interfaces."""

import asyncio

from databricks_ai_bridge.durable_runtime import DurableExecutionContext, JsonObject


async def run_agent(request: JsonObject, context: DurableExecutionContext) -> JsonObject:
    payload = request["payload"]
    steps = int(payload.get("steps", 3))
    delay_seconds = float(payload.get("delay_seconds", 1))

    for step in range(1, steps + 1):
        await asyncio.sleep(delay_seconds)
        await context.emit({"type": "progress", "step": step, "total": steps})

    return {
        "message": "completed",
        "session_id": request["session_id"],
        "attempt": context.attempt,
    }
