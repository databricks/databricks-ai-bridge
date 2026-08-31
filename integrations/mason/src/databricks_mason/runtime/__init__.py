"""Agent-side runtime helpers a deployed Mason agent imports (installed via ``databricks-mason[runtime]``).

The session-store checkpointer, MLflow tracing setup, MCP tool loading, background-run store,
durability log, and recovery workflow. These need the agent stack (databricks-langchain, langgraph,
langchain, fastapi, mlflow), so they live behind the ``[runtime]`` optional extra to keep the CLI
install light. Agents import from here rather than vendoring copies; the CLI itself does not import
this subpackage.
"""
