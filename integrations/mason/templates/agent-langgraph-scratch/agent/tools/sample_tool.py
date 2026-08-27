"""Sample LangChain tool."""

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()
