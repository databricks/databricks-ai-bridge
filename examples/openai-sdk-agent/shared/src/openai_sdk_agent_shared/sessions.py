"""OpenAI Agents SDK sessions stored in the App's Lakebase resource."""

import os

from databricks_openai.agents import AsyncDatabricksSession


def create_session(session_id: str) -> AsyncDatabricksSession:
    return AsyncDatabricksSession(
        session_id=session_id,
        autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"],
        schema=os.getenv("LAKEBASE_SESSION_SCHEMA", "openai_sdk_agent_sessions"),
    )
