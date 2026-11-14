"""Serve a framework-neutral agent with the managed invocation protocol."""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from databricks_agentkit import DurableAgentServer, configure_tracing
from runtime.adapter import invoke, recover

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
configure_tracing()

app = DurableAgentServer()
app.invoke(invoke)
app.recover(recover)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
