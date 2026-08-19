"""OpenAI Agents SDK session history stored in the App's Lakebase resource."""

import os

from databricks_openai.agents import AsyncDatabricksSession


def create_session(session_id: str) -> AsyncDatabricksSession:
    return AsyncDatabricksSession(
        session_id=session_id,
        autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"],
        schema=os.getenv("LAKEBASE_SESSION_SCHEMA", "openai_sdk_agent_sessions"),
    )


async def initialize_sessions() -> None:
    session = create_session("__startup__")
    await session._ensure_tables()
