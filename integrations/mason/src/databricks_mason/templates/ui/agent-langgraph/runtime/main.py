"""Run the LangGraph agent with the optional Mason chat app."""

import os
from pathlib import Path

import uvicorn
from agent.agent import configure
from dotenv import load_dotenv
from runtime.adapter import invoke, recover
from runtime.ui import install_ui

from databricks_mason import AgentApp
from databricks_mason.runtime.auth import InvocationAuthPolicy

# override=False so injected DATABRICKS_* (from `mason dev -p` or the deploy platform) win over a
# checked-in .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
configure()

app = AgentApp(auth_policy=InvocationAuthPolicy.from_manifest())
app.invoke(invoke)
if not app.auth_policy.requires_user:
    app.recover(recover)
install_ui(app)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
