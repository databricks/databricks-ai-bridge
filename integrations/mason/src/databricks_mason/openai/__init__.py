"""OpenAI Agents SDK adapter for running an agent on Databricks.

Composable pieces you drop into an existing OpenAI Agents SDK agent include explicit Databricks
integration binding, session storage, long-term memory tools, and MLflow tracing. Install them via
``databricks-mason[runtime-openai]``::

    from contextlib import AsyncExitStack

    from databricks_mason.openai import bind_tools, memory_tools, session_store

    async with AsyncExitStack() as stack:
        agent = Agent(
            name="Agent",
            model="databricks-gpt-5-2",
            tools=[*your_tools, *memory_tools()],
        )
        agent = await bind_tools(agent, DATABRICKS_TOOLS, stack=stack)
        result = await Runner.run(agent, messages, session=session_store(session_id))

``__all__`` is the curated surface. Other entry points (``DatabricksSessionStore``) are
reachable by their submodule paths but not re-exported here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_mason.openai.mcp import bind_tools
    from databricks_mason.openai.memory import memory_tools, recall, remember
    from databricks_mason.openai.sessions import session_store
    from databricks_mason.runtime import tag_session, workspace_client, workspace_headers


def configure_tracing() -> None:
    """Enable MLflow tracing with OpenAI autologging. Call once at startup.

    Safe to call unconditionally — tracing turns on only when the MLflow destination and experiment
    are configured in the environment (see :func:``databricks_mason.runtime.configure_tracing``).
    """
    import mlflow

    from databricks_mason.runtime import configure_tracing as _configure_tracing

    _configure_tracing(autolog=mlflow.openai.autolog)


__all__ = [
    # Explicit Databricks integrations — bind them for the lifetime of the caller-owned stack.
    "bind_tools",
    # Long-term memory tools (opt-in via AGENT_MEMORY_STORE) — add to your tool list.
    "memory_tools",
    "remember",
    "recall",
    # Session persistence — pass session_store(session_id) to Runner.run(session=...).
    "session_store",
    # MLflow tracing (OpenAI autolog bound in) — call configure_tracing() once at startup.
    "configure_tracing",
    "tag_session",
    # Workspace SDK client construction.
    "workspace_client",
    "workspace_headers",
]

# Re-exports resolved lazily (PEP 562) so importing one submodule (e.g. ``.mcp``) does not eagerly
# pull in the others' dependencies. ``configure_tracing`` is defined above (binds OpenAI autolog).
_MODULE_BY_NAME = {
    "bind_tools": "databricks_mason.openai.mcp",
    "memory_tools": "databricks_mason.openai.memory",
    "remember": "databricks_mason.openai.memory",
    "recall": "databricks_mason.openai.memory",
    "session_store": "databricks_mason.openai.sessions",
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
