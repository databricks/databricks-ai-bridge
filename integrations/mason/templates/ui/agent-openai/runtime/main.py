"""Agent server entry point with the optional Mason chat app installed."""

import os
from pathlib import Path

import agent.agent
import uvicorn
from dotenv import load_dotenv

from runtime.runtime import build_app
from runtime.ui import install_ui

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
agent.agent.configure()

app = build_app(agent.agent.invoke_handler, agent.agent.stream_handler)
install_ui(app)


def main():
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
