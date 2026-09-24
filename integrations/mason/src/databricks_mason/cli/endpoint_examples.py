"""Copy-pasteable examples for invoking Agent Bricks-generated agent endpoints."""

from __future__ import annotations

import shlex

from rich.text import Text

from databricks_mason import render


def agent_invoke_command(target: str, *, uses_runtime_api: bool) -> str:
    """Build a single-line example, generating a fresh runtime invocation ID on every run."""
    path = "/api/invocations" if uses_runtime_api else "/invocations"
    body = '{"input":[{"role":"user","content":"hi"}]}'
    if uses_runtime_api:
        # Double quotes allow command substitution while preserving the JSON's literal quotes.
        body = '{"id":"$(uuidgen)",' + body[1:]
        json_arg = '"' + body.replace('"', r"\"") + '"'
    else:
        json_arg = shlex.quote(body)
    return f"ab endpoint invoke {target} --path {path} --json {json_arg}"


def print_agent_invoke_command(target: str, *, uses_runtime_api: bool) -> None:
    """Print only the new invocation example without changing the existing success panel."""
    con = render.console()
    con.print(Text("Invoke with Agent Bricks", style=render.SECONDARY))
    # A separate, soft-wrapped line avoids copying panel borders or inserted line breaks.
    con.print(
        Text(agent_invoke_command(target, uses_runtime_api=uses_runtime_api), style=render.COMMAND),
        soft_wrap=True,
    )
