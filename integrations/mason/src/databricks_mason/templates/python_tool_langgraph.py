"""A Mason-scaffolded LangGraph tool."""

from langchain_core.tools import tool  # ty: ignore[unresolved-import]


@tool
def __MASON_TOOL_FUNCTION__(value: str) -> str:
    """Implement __MASON_TOOL_FUNCTION__."""
    raise NotImplementedError("Implement __MASON_TOOL_FUNCTION__")
