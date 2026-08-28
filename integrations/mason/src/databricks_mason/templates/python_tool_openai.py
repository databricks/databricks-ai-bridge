"""A Mason-scaffolded OpenAI Agents SDK tool."""

from agents import function_tool  # ty: ignore[unresolved-import]


@function_tool
def __MASON_TOOL_FUNCTION__(value: str) -> str:
    """Implement __MASON_TOOL_FUNCTION__."""
    raise NotImplementedError("Implement __MASON_TOOL_FUNCTION__")
