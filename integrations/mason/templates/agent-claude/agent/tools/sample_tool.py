"""Sample tool. A working example — add your own tools as new files in this package.

Decorate a function with ``@beta_tool`` (Anthropic Tool Runner) and it becomes an agent tool; the
package auto-collects it via ``all_tools()``, which ``create_tools`` uses.
"""

from datetime import datetime

from anthropic import beta_tool


@beta_tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()
