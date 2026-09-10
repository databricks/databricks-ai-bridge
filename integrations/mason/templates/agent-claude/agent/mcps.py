"""MCP servers to offer the agent — this is where you configure them.

Empty by default: the agent runs with only the MCP servers declared in ``agent.toml``. Add
MCP-connector server dicts here to offer more; ``agent.py`` joins them with the ``agent.toml``
servers and passes them to the Tool Runner.
"""


def build_mcp_servers() -> list[dict]:
    """Return extra Anthropic MCP-connector servers to offer the agent. Empty by default.

    Example::

        return [{"type": "url", "name": "docs", "url": "https://mcp.example.com/sse"}]
    """
    return []
