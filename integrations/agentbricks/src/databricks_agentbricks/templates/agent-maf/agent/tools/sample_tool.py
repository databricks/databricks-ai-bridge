"""Sample tool. A working example — add your own tools as new files in this package.

Decorate a function with ``@tool`` (Microsoft Agent Framework) and it becomes an agent tool; the
package auto-collects it via ``all_tools()``, which ``create_agent`` uses.
"""

from datetime import datetime

from agent_framework import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()
