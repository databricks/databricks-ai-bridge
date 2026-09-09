"""Run the OpenAI Agents SDK agent through Mason's durable application."""

import os
from pathlib import Path

# Importing the agent is side-effect-free (no env is read until configure()), so it sits up top.
import agent.agent
import uvicorn
from dotenv import load_dotenv

from databricks_mason import AgentApp

# override=False so injected DATABRICKS_* (from `mason dev -p` or the deploy platform) win over a
# checked-in .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
agent.agent.configure()

app = AgentApp()
app.invoke(agent.agent.invoke)
if app.durable_runtime:
    app.on_recovery(agent.agent.on_recovery)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
