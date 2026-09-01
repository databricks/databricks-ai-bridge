"""LangGraph adapter for running an agent on Databricks (installed via ``databricks-mason[runtime]``).

Composable pieces you drop into an existing LangGraph agent — a session-store checkpointer, MCP
tools declared in ``agent.toml``, long-term memory tools, and MLflow tracing. Each maps onto a slot
LangGraph already has, so migrating an existing agent is a graft, not a rewrite::

    from databricks_mason.langgraph import (
        checkpointer,
        thread_config,
        mcp_tools,
        configure_tracing,
    )

    configure_tracing()
    agent = create_agent(
        model=...,
        tools=[
            *your_tools,
            *await mcp_tools(),
        ],  # mcp_tools() includes your agent.toml servers
        checkpointer=checkpointer(),  # durable when AGENT_SESSION_STORE is set
    )
    result = await agent.ainvoke(inputs, config=thread_config(session_id))

These need the agent stack (databricks-langchain, langgraph, langchain, fastapi, mlflow), so they sit
behind the ``[runtime]`` extra to keep a plain ``databricks-mason`` CLI install light.

``__all__`` is the curated surface. Other entry points (``mcp_client``, ``DatabricksSessionStoreSaver``,
and the ``recovery`` / ``long_running`` stop-start demo modules) are reachable by their submodule
paths but not re-exported here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_mason.langgraph.mcp import mcp_tools
    from databricks_mason.langgraph.memory import memory_tools, recall, remember
    from databricks_mason.langgraph.session_store import checkpointer, thread_config
    from databricks_mason.runtime import tag_session, workspace_client, workspace_headers


def configure_tracing() -> None:
    """Enable MLflow tracing with LangChain autologging. Call once at startup.

    Safe to call unconditionally — tracing turns on only when the MLflow destination and experiment
    are configured in the environment (see :func:`databricks_mason.runtime.configure_tracing`).
    """
    import mlflow

    from databricks_mason.runtime import configure_tracing as _configure_tracing

    _configure_tracing(autolog=mlflow.langchain.autolog)


__all__ = [
    # MCP tools from agent.toml (plus any servers you pass) — add them to your agent's tool list.
    "mcp_tools",
    # Long-term memory tools (opt-in via AGENT_MEMORY_STORE) — add to your tool list.
    "memory_tools",
    "remember",
    "recall",
    # Session persistence — pass checkpointer() to create_agent(checkpointer=...) and
    # thread_config(session_id) as the per-request run config.
    "checkpointer",
    "thread_config",
    # MLflow tracing (LangChain autolog bound in) — call configure_tracing() once at startup.
    "configure_tracing",
    "tag_session",
    # Workspace SDK client construction.
    "workspace_client",
    "workspace_headers",
]

# Re-exports resolved lazily (PEP 562) so importing one submodule (e.g. ``.mcp``) does not eagerly
# pull in the others' dependencies. ``configure_tracing`` is defined above (binds LangChain autolog).
_MODULE_BY_NAME = {
    "mcp_tools": "databricks_mason.langgraph.mcp",
    "memory_tools": "databricks_mason.langgraph.memory",
    "remember": "databricks_mason.langgraph.memory",
    "recall": "databricks_mason.langgraph.memory",
    "checkpointer": "databricks_mason.langgraph.session_store",
    "thread_config": "databricks_mason.langgraph.session_store",
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
