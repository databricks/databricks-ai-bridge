"""Minimal LangGraph workload for the Mason durability template."""

import asyncio
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from databricks_mason import DurableAgentContext


class AgentState(TypedDict):
    message: str
    delay_seconds: float
    result: NotRequired[str]


async def _process(state: AgentState) -> dict[str, str]:
    await asyncio.sleep(state["delay_seconds"])
    return {"result": f"Processed: {state['message']}"}


builder = StateGraph(AgentState)
builder.add_node("process", _process)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()


async def run_agent(payload: dict, context: DurableAgentContext) -> dict:
    agent_input = payload.get("input", {})
    if not isinstance(agent_input, dict):
        raise ValueError("input must be an object")

    message = agent_input.get("message", "hello")
    delay_seconds = agent_input.get("delay_seconds", 0)
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    if isinstance(delay_seconds, bool) or not isinstance(delay_seconds, (int, float)):
        raise ValueError("delay_seconds must be a number")
    if delay_seconds < 0 or delay_seconds > 300:
        raise ValueError("delay_seconds must be between 0 and 300")

    await context.emit(
        {
            "type": "progress",
            "stage": "recovered" if context.is_recovery else "started",
            "attempt": context.attempt,
        }
    )
    result = await graph.ainvoke({"message": message, "delay_seconds": float(delay_seconds)})
    await context.emit({"type": "progress", "stage": "completed", "attempt": context.attempt})
    return {
        "result": result["result"],
        "attempt": context.attempt,
        "recovered": context.is_recovery,
    }
