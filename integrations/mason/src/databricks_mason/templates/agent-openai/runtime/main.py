"""Run the OpenAI Agents SDK agent through Mason Runtime."""

import os
from pathlib import Path

# Importing the agent is side-effect-free (no env is read until configure()), so it sits up top.
import uvicorn
from agent.agent import configure
from dotenv import load_dotenv

from databricks_mason import AgentApp
from databricks_mason.runtime.auth import InvocationAuthPolicy
from runtime.adapter import invoke, recover

# override=False so injected DATABRICKS_* (from `mason dev -p` or the deploy platform) win over a
# checked-in .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
configure()

app = AgentApp(auth_policy=InvocationAuthPolicy.from_manifest())
app.invoke(invoke)
if not app.auth_policy.requires_user:
    app.recover(recover)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
