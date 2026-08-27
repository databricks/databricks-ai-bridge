"""OpenAI Agents SDK loop for durable streaming and human approval."""

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any

from agents import Agent, Runner, function_tool
from databricks_openai.agents import AsyncDatabricksSession

EventEmitter = Callable[[dict[str, Any]], Awaitable[int]]


@function_tool
async def complete_approved_action(action: str, wait_seconds: int) -> str:
    """Simulate an approved side effect after a bounded delay."""
    if wait_seconds < 0 or wait_seconds > 300:
        raise ValueError("wait_seconds must be between 0 and 300")
    await asyncio.sleep(wait_seconds)
    return f"Completed approved action: {action}"


def _create_agent() -> Agent:
    return Agent(
        name="Durable approval assistant",
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        instructions=(
            "For a PROPOSAL request, produce a short plan and do not call tools. "
            "For an APPROVED request, always call complete_approved_action. "
            "For a REJECTED request, acknowledge the rejection without calling tools."
        ),
        tools=[complete_approved_action],
    )


async def run_openai_agent(
    payload: dict[str, Any],
    session_id: str,
    emit: EventEmitter,
    *,
    is_recovery: bool = False,
) -> dict[str, Any]:
    session = AsyncDatabricksSession(
        session_id=session_id,
        autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"],
        schema=os.getenv(
            "OPENAI_AGENT_SESSION_SCHEMA", "durable_openai_agent_sessions"
        ),
    )
    action = str(payload["action"])
    decision = payload.get("decision")
    wait_seconds = int(payload.get("wait_seconds", 0))
    if decision is None:
        instruction = f"PROPOSAL: Draft a plan for this action: {action}"
    elif decision == "approve":
        instruction = (
            f"APPROVED: Complete this action: {action}. "
            f"Call complete_approved_action with wait_seconds={wait_seconds}."
        )
    elif decision == "reject":
        instruction = f"REJECTED: Do not perform this action: {action}"
    else:
        raise ValueError("decision must be approve, reject, or omitted")

    if is_recovery:
        instruction = (
            "RECOVERY: Continue the interrupted durable run using the existing session. "
            + instruction
        )

    result = Runner.run_streamed(_create_agent(), input=instruction, session=session)

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            await emit(event.data.model_dump(mode="json"))

    output = str(result.final_output or "")
    if decision is None:
        response = {"status": "requires_action", "action": action, "proposal": output}
        await emit({"type": "agent.approval_required", "action": action})
        return response

    status = "completed" if decision == "approve" else "rejected"
    await emit({"type": f"agent.{status}", "action": action})
    return {"status": status, "action": action, "output": output}
