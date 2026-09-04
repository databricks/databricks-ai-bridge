"""Restart-safe Databricks App entry point for both recovery strategies."""

import os

import uvicorn

from databricks_ai_bridge.long_running import ResumeStrategy
from langgraph_agent_shared.runtime import create_app


def _resume_strategy() -> ResumeStrategy:
    configured = os.getenv("RESUME_STRATEGY")
    if configured:
        return ResumeStrategy(configured)
    app_name = os.getenv("DATABRICKS_APP_NAME")
    app_strategies = {
        "langgraph-event-recovery": ResumeStrategy.EVENT_LOG,
        "langgraph-session-recovery": ResumeStrategy.AGENT_SESSION,
    }
    if app_name in app_strategies:
        return app_strategies[app_name]
    raise ValueError("Set RESUME_STRATEGY when running outside the cookbook Apps")


resume_strategy = _resume_strategy()
if resume_strategy is ResumeStrategy.AGENT_SESSION:
    os.environ.setdefault(
        "LAKEBASE_CHECKPOINT_SCHEMA",
        "langgraph_agent_session_checkpoints",
    )
else:
    os.environ.setdefault(
        "LAKEBASE_CHECKPOINT_SCHEMA",
        "langgraph_event_log_checkpoints",
    )

import langgraph_agent_shared.handlers  # noqa: E402,F401

app = create_app(resume_strategy)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
