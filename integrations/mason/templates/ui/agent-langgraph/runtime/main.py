"""Run the SDK-hosted durable agent application with the optional Mason UI."""

import os
from pathlib import Path

import agent.agent
import uvicorn
from databricks_mason import DurableAgentApp
from dotenv import load_dotenv

from runtime.ui import install_ui

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
agent.agent.configure()

server = DurableAgentApp(
    agent.agent.invoke,
    on_resume=agent.agent.recover,
)
app = server.app
install_ui(app)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
