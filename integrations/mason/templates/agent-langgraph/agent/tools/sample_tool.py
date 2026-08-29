"""Sample manifest-activated tool.

Decorated Python is executable tool code, but it is offered to the agent only when ``agent.toml``
declares its exact ``module:attribute`` entry point. Mason initializes this sample's declaration.
"""

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()
