"""Copy-pasteable examples for invoking Mason-generated agent endpoints."""

from __future__ import annotations

import shlex


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
    return f"mason endpoint invoke {target} --path {path} --json {json_arg}"
