"""Run the OpenAI Agents SDK agent with the optional Mason chat app."""

import os
from pathlib import Path

import uvicorn
from agent.agent import configure
from dotenv import load_dotenv
from runtime.adapter import invoke, recover
from runtime.ui import install_ui

from databricks_agentkit import DurableAgentServer

# override=False so injected DATABRICKS_* (from `ab dev -p` or the deploy platform) win over a
# checked-in .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
configure()

app = DurableAgentServer()
app.invoke(invoke)
app.recover(recover)
install_ui(app)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
