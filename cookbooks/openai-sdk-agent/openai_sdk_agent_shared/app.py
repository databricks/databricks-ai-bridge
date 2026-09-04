"""Restart-safe Databricks App entry point for both recovery strategies."""

import os

import uvicorn

from databricks_ai_bridge.long_running import LongRunningAgentServer, ResumeStrategy


def _resume_strategy() -> ResumeStrategy:
    configured = os.getenv("RESUME_STRATEGY")
    if configured:
        return ResumeStrategy(configured)
    app_name = os.getenv("DATABRICKS_APP_NAME")
    app_strategies = {
        "openai-event-log-recovery": ResumeStrategy.EVENT_LOG,
        "openai-agent-session": ResumeStrategy.AGENT_SESSION,
    }
    if app_name in app_strategies:
        return app_strategies[app_name]
    raise ValueError("Set RESUME_STRATEGY when running outside the cookbook Apps")


resume_strategy = _resume_strategy()
if resume_strategy is ResumeStrategy.AGENT_SESSION:
    os.environ.setdefault("LAKEBASE_SESSION_SCHEMA", "openai_agent_session_sessions")
else:
    os.environ.setdefault("LAKEBASE_SESSION_SCHEMA", "openai_event_log_sessions")

import openai_sdk_agent_shared.handlers  # noqa: E402,F401

agent_server = LongRunningAgentServer(
    "ResponsesAgent",
    db_autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"],
    resume_strategy=resume_strategy,
)
app = agent_server.app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
