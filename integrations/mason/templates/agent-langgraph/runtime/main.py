"""Run the LangGraph agent through Mason's durable application."""

import os
from pathlib import Path

# Importing the agent is side-effect-free (no env is read until configure()), so it sits up top.
import agent.agent
import uvicorn
from dotenv import load_dotenv

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


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
