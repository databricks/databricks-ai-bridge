"""Run the SDK-hosted durable agent application."""

import os
from pathlib import Path

import agent.agent
import uvicorn
from databricks_mason import DurableAgentApp
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
agent.agent.configure()

server = DurableAgentApp(
    agent.agent.invoke,
    on_resume=agent.agent.recover,
)
app = server.app


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
