"""Run the durable OpenAI Agents SDK agent with the optional Mason chat app."""

import os
from pathlib import Path

import agent.agent
import uvicorn
from dotenv import load_dotenv
from runtime.ui import install_ui

from databricks_mason import AgentApp

# .env fills unset config only; the real environment wins (override=False). `mason dev -p` and
# the deploy platform inject DATABRICKS_* into the process, and a checked-in .env must not clobber
# them — overriding only the profile while leaving the injected host mismatches host and credential.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
agent.agent.configure()

DURABLE_RUNTIME = True

app = AgentApp(durable_runtime=DURABLE_RUNTIME)
app.invoke(agent.agent.invoke)
if DURABLE_RUNTIME:
    app.on_recovery(agent.agent.on_recovery)
install_ui(app)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
