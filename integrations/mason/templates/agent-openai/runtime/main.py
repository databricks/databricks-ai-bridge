"""Run the OpenAI Agents SDK agent through Mason's durable application."""

import os
from pathlib import Path

# Importing the agent is side-effect-free (no env is read until configure()), so it sits up top.
import agent.agent
import uvicorn
from dotenv import load_dotenv

from databricks_mason import DurableAgentApp, auto_recovery_enabled

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
agent.agent.configure()

app = DurableAgentApp()
app.invoke(agent.agent.invoke)
if auto_recovery_enabled():
    app.on_recovery(agent.agent.on_recovery)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
