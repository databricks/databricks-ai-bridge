"""OpenAI Agents SDK adapter for running an agent on Databricks (installed via ``databricks-agentbricks[openai]``).

Composable pieces you drop into an existing OpenAI Agents SDK agent — a session store, MCP servers
declared in ``agent.toml``, long-term memory tools, and MLflow tracing. Each maps onto a slot the
Agents SDK already has, so migrating an existing agent is a graft, not a rewrite::

    from databricks_agentbricks.openai import (
        session_store,
        mcp_servers,
        memory_tools,
        configure_tracing,
    )

    configure_tracing()
    agent = Agent(
        name="Agent",
        model="databricks-gpt-5-2",
        tools=[*your_tools, *memory_tools()],
        mcp_servers=await mcp_servers(),  # your agent.toml servers + any you pass
    )
    result = await Runner.run(agent, messages, session=session_store(session_id))

These need the agent stack (openai-agents, databricks-openai, mlflow), so they sit behind the
``[openai]`` extra to keep a plain ``databricks-agentbricks`` installation independent of agent frameworks.

``__all__`` is the curated surface. Other entry points (``DatabricksSessionStore``) are reachable by
their submodule paths but not re-exported here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_agentbricks.openai.genie import genie_tools
    from databricks_agentbricks.openai.mcp import mcp_servers
    from databricks_agentbricks.openai.memory import memory_tools
    from databricks_agentbricks.openai.sessions import session_store
    from databricks_agentbricks.runtime import (
        start_trace,
        workspace_client,
        workspace_headers,
    )


def configure_tracing() -> None:
    """Enable MLflow tracing with OpenAI autologging. Call once at startup.

    Safe to call unconditionally — tracing turns on only when the MLflow destination and experiment
    are configured in the environment (see :func:`databricks_agentbricks.runtime.configure_tracing`).
    """
    import mlflow

    from databricks_agentbricks.runtime import configure_tracing as _configure_tracing

    _configure_tracing(autolog=mlflow.openai.autolog)


__all__ = [
    "genie_tools",
    # MCP servers from agent.toml (plus any you pass) — hand them to Agent(mcp_servers=...).
    "mcp_servers",
    # Long-term memory tools (opt-in via AGENT_MEMORY_STORE) — add to your tool list.
    "memory_tools",
    # Session persistence — pass session_store(session_id) to Runner.run(session=...).
    "session_store",
    # MLflow tracing (OpenAI autolog bound in) — call configure_tracing() once at startup.
    "configure_tracing",
    # Wrap each invocation in start_trace() so a trace is recorded (pass session_id= to tag it).
    "start_trace",
    # Workspace SDK client construction.
    "workspace_client",
    "workspace_headers",
]

# Re-exports resolved lazily (PEP 562) so importing one submodule (e.g. ``.mcp``) does not eagerly
# pull in the others' dependencies. ``configure_tracing`` is defined above (binds OpenAI autolog).
_MODULE_BY_NAME = {
    "genie_tools": "databricks_agentbricks.openai.genie",
    "mcp_servers": "databricks_agentbricks.openai.mcp",
    "memory_tools": "databricks_agentbricks.openai.memory",
    "session_store": "databricks_agentbricks.openai.sessions",
    "start_trace": "databricks_agentbricks.runtime",
    "workspace_client": "databricks_agentbricks.runtime",
    "workspace_headers": "databricks_agentbricks.runtime",
}


def __getattr__(name: str) -> object:
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
