"""Run the SDK-hosted durable agent application."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
from agent.agent import app, configure  # noqa: E402


def main() -> None:
    configure()
    app.run()
