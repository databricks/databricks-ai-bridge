"""A Mason-scaffolded LangGraph tool."""

# mason:python-tool id=__MASON_TOOL_ID__ entrypoint=agent.tools.__MASON_TOOL_MODULE__:__MASON_TOOL_FUNCTION__

from langchain_core.tools import tool  # ty: ignore[unresolved-import]


@tool
def __MASON_TOOL_FUNCTION__(value: str) -> str:
    """Implement __MASON_TOOL_FUNCTION__."""
    raise NotImplementedError("Implement __MASON_TOOL_FUNCTION__")
