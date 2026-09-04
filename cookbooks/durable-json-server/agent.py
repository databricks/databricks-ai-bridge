"""OpenAI Agents SDK loop hosted by the generic durable server."""

from openai_agent import run_openai_agent

from databricks_ai_bridge.durable_server import (
    DatabricksDurableServer,
    DurableRequestContext,
)


async def agent(payload: dict, context: DurableRequestContext) -> dict:
    result = await run_openai_agent(
        payload=payload,
        session_id=context.session_id,
        emit=context.emit,
    )

    return {
        "result": result,
        "session_id": context.session_id,
        "attempt": context.attempt,
    }


async def resume_agent(payload: dict, context: DurableRequestContext) -> dict:
    result = await run_openai_agent(
        payload=payload,
        session_id=context.session_id,
        emit=context.emit,
        is_recovery=True,
    )
    return {
        "result": result,
        "session_id": context.session_id,
        "attempt": context.attempt,
    }


server = DatabricksDurableServer(agent, on_resume=resume_agent)
app = server.app
