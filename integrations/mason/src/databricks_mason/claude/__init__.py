"""Anthropic SDK Tool Runner adapter for running an agent on Databricks (installed via ``databricks-mason[runtime-claude]``).

Composable pieces you drop into an agent built on the Anthropic Python SDK's Tool Runner
(``client.beta.messages.tool_runner`` / ``@beta_tool``) — a conversation-history helper, MCP servers
declared in ``agent.toml``, long-term memory tools, MLflow tracing, and an ``anthropic`` client that
picks up ``ANTHROPIC_API_KEY`` (or Bedrock)::

    from databricks_mason.claude import (
        client,
        configure_tracing,
        mcp_servers,
        memory_tools,
        session_history,
    )

    configure_tracing()
    history = session_history(session_id, actor)
    runner = client().beta.messages.tool_runner(
        model="claude-opus-5",
        max_tokens=16000,
        tools=[*your_tools, *memory_tools(actor)],
        messages=[*history.load(), *new_messages],
    )

These need the agent stack (anthropic, mlflow), so they sit behind the ``[runtime-claude]`` extra to
keep a plain ``databricks-mason`` CLI install light. ``__all__`` is the curated surface.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_mason.claude.client import client
    from databricks_mason.claude.mcp import mcp_servers
    from databricks_mason.claude.memory import memory_tools
    from databricks_mason.claude.sessions import session_history
    from databricks_mason.runtime import tag_session, workspace_client, workspace_headers


def configure_tracing() -> None:
    """Enable MLflow tracing with Anthropic autologging. Call once at startup.

    Safe to call unconditionally — tracing turns on only when the MLflow destination and experiment
    are configured in the environment (see :func:`databricks_mason.runtime.configure_tracing`).
    """
    import mlflow

    from databricks_mason.runtime import configure_tracing as _configure_tracing

    _configure_tracing(autolog=mlflow.anthropic.autolog)


__all__ = [
    # An `anthropic` client (Anthropic, or AnthropicBedrock when CLAUDE_CODE_USE_BEDROCK is set).
    "client",
    # MCP servers from agent.toml (plus any you pass) — hand to the runner as `mcp_servers=`.
    "mcp_servers",
    # Long-term memory tools (opt-in via AGENT_MEMORY_STORE) — add to your tool list.
    "memory_tools",
    # Conversation history — the Tool Runner is stateless, so load()/append() the transcript.
    "session_history",
    # MLflow tracing (Anthropic autolog bound in) — call configure_tracing() once at startup.
    "configure_tracing",
    "tag_session",
    "workspace_client",
    "workspace_headers",
]

_MODULE_BY_NAME = {
    "client": "databricks_mason.claude.client",
    "mcp_servers": "databricks_mason.claude.mcp",
    "memory_tools": "databricks_mason.claude.memory",
    "session_history": "databricks_mason.claude.sessions",
    "tag_session": "databricks_mason.runtime",
    "workspace_client": "databricks_mason.runtime",
    "workspace_headers": "databricks_mason.runtime",
}


def __getattr__(name: str) -> object:
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
