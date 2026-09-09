"""Run the durable OpenAI Agents SDK agent with the optional Mason chat app."""

import os
from pathlib import Path

import agent.agent
import uvicorn
from dotenv import load_dotenv
from runtime.ui import install_ui

from databricks_mason import AgentApp
from databricks_mason.agent_project import AgentProject

# override=False so injected DATABRICKS_* (from `mason dev -p` or the deploy platform) win over a
# checked-in .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
agent.agent.configure()

durable_runtime = AgentProject.load(Path(__file__).parent.parent).durability_enabled
app = AgentApp(durable_runtime=durable_runtime)
app.invoke(agent.agent.invoke)
if app.durable_runtime:
    app.on_recovery(agent.agent.on_recovery)
install_ui(app)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
