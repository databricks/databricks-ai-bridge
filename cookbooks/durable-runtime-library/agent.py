"""OpenAI Agents SDK adapter for the transport-neutral durable runtime."""

from openai_agent import run_openai_agent

from databricks_ai_bridge.durable_runtime import DurableExecutionContext, JsonObject


async def run_agent(request: JsonObject, context: DurableExecutionContext) -> JsonObject:
    payload = request["payload"]
    session_id = str(request["session_id"])
    output = await run_openai_agent(
        prompt=str(payload["prompt"]),
        session_id=session_id,
        emit=context.emit,
    )

    return {
        "output": output,
        "session_id": session_id,
        "attempt": context.attempt,
    }
