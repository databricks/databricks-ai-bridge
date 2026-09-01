"""OpenAI Agents SDK adapter for the transport-neutral durable runtime."""

from openai_agent import run_openai_agent

from databricks_ai_bridge.durable_runtime import DurableExecutionContext, JsonObject


async def run_agent(
    request: JsonObject, context: DurableExecutionContext
) -> JsonObject:
    payload = request["payload"]
    session_id = str(request["session_id"])
    result = await run_openai_agent(
        payload=payload,
        session_id=session_id,
        emit=context.emit,
        is_recovery=context.is_recovery,
    )

    return {
        "result": result,
        "session_id": session_id,
        "attempt": context.attempt,
    }
