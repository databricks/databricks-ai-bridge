"""Run the OpenAI Agents SDK agent through Agent Bricks Runtime."""

import os
from pathlib import Path

# Importing the agent is side-effect-free (no env is read until configure()), so it sits up top.
import uvicorn
from agent.agent import configure
from dotenv import load_dotenv

from databricks_agentkit import DurableAgentServer
from runtime.adapter import invoke, recover

# override=False so injected DATABRICKS_* (from `ab dev -p` or the deploy platform) win over a
# checked-in .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
configure()

app = DurableAgentServer()
app.invoke(invoke)
app.recover(recover)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
