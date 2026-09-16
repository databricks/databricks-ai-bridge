"""Copy the emitted command unchanged into real shells and check the resulting arguments."""

from __future__ import annotations

import io
import json
import shlex
import shutil
import subprocess
import uuid

import pytest
from rich.console import Console

from databricks_mason import render
from databricks_mason.cli.endpoint_examples import agent_invoke_command


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("width", [40, 80, 200])
@pytest.mark.parametrize("target", ["--url http://localhost:8000", "agent-mason-myapp"])
@pytest.mark.parametrize("uses_runtime_api", [False, True])
def test_printed_invoke_command_is_shell_copyable(shell, width, target, uses_runtime_api):
    executable = shutil.which(shell)
    if executable is None or (uses_runtime_api and shutil.which("uuidgen") is None):
        pytest.skip(f"requires {shell} and uuidgen for runtime examples")

    buf = io.StringIO()
    render.success(
        "Ready",
        next_steps=[
            (agent_invoke_command(target, uses_runtime_api=uses_runtime_api), "Send a test request")
        ],
        con=Console(file=buf, width=width, no_color=True),
    )
    # Copy the whole line, without removing borders, stripping padding, or joining wrapped lines.
    commands = [line for line in buf.getvalue().splitlines() if line.startswith("mason endpoint")]
    assert len(commands) == 1, buf.getvalue()
    command = commands[0]
    path = "/api/invocations" if uses_runtime_api else "/invocations"
    ids = set()
    for _ in range(2):
        # Capture argv instead of reaching a workspace; functional tests exercise the real CLI/HTTP.
        result = subprocess.run(
            [executable, "-f", "-c", 'mason() { printf "%s\\0" "$@"; }\n' + command],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        args = result.stdout.removesuffix("\0").split("\0")
        assert args[:-1] == ["endpoint", "invoke", *shlex.split(target), "--path", path, "--json"]
        body = json.loads(args[-1])
        assert body.pop("input") == [{"role": "user", "content": "hi"}]
        if uses_runtime_api:
            invocation_id = uuid.UUID(body.pop("id"))
            assert invocation_id not in ids
            ids.add(invocation_id)
        assert body == {}
