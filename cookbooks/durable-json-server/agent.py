"""OpenAI Agents SDK loop hosted by the generic durable server."""

from openai_agent import run_openai_agent

from databricks_ai_bridge.durable_server import (
    DatabricksDurableServer,
    DurableRequestContext,
)


async def agent(payload: dict, context: DurableRequestContext) -> dict:
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


server = DatabricksDurableServer(agent)
app = server.app
