"""Run the SDK-hosted durable agent application with the optional Mason UI."""

from pathlib import Path

from dotenv import load_dotenv

from runtime.ui import install_ui

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
from agent.agent import app, configure  # noqa: E402

install_ui(app)


def main() -> None:
    configure()
    app.run()
