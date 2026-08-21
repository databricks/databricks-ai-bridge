"""OpenAI agent using its SDK session for recovery."""

import os

import handlers  # noqa: F401
import uvicorn

from databricks_ai_bridge.long_running import LongRunningAgentServer, ResumeStrategy

agent_server = LongRunningAgentServer(
    "ResponsesAgent",
    db_autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"],
    resume_strategy=ResumeStrategy.AGENT_SESSION,
)
app = agent_server.app


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
