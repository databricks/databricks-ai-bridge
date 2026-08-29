"""Agent tools explicitly activated by the project's ``agent.toml``."""

from agent.mason.python_runtime import python_tools


def all_tools():
    """Return only Python tools explicitly activated in ``agent.toml``."""
    return python_tools()
