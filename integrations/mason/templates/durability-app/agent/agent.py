"""Minimal LangGraph workload for the Mason durability template."""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from databricks_mason import DurableAgentContext


class AgentState(TypedDict):
    message: str
    result: NotRequired[str]


def _process(state: AgentState) -> dict[str, str]:
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

    message = agent_input.get("message")
    if not isinstance(message, str):
        raise ValueError("message must be a string")

    await context.emit(
        {
            "type": "progress",
            "stage": "recovered" if context.is_recovery else "started",
            "attempt": context.attempt,
        }
    )
    result = await graph.ainvoke({"message": message})
    await context.emit({"type": "progress", "stage": "completed", "attempt": context.attempt})
    return {
        "result": result["result"],
        "attempt": context.attempt,
        "recovered": context.is_recovery,
    }
