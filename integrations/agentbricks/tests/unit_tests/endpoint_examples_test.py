"""Copy the emitted command from the success panel into real shells and check the arguments."""

from __future__ import annotations

import io
import json
import shlex
import shutil
import subprocess
import uuid

import pytest
from rich.console import Console

from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.endpoint import (
    agent_invoke_command,
    agent_invoke_step,
)

_TARGETS = ["--url http://localhost:8000", "agent-bricks-myapp"]
_STEPS = [("agentbricks dev", "Run locally"), ("agentbricks deploy my-agent", "Deploy it")]


def _panel_rows(output: str) -> list[str]:
    """The panel's content rows, with the `│ ` / ` │` borders and right padding stripped."""
    return [
        line.removeprefix("│ ").removesuffix("│").rstrip()
        for line in output.splitlines()
        if line.startswith("│")
    ]


def _render(width: int, target: str, *, uses_runtime_api: bool, terminal: bool = False) -> str:
    buf = io.StringIO()
    con = Console(file=buf, width=width, no_color=True, force_terminal=terminal)
    render.success(
        "Started",
        next_steps=[*_STEPS, agent_invoke_step(target, uses_runtime_api=uses_runtime_api)],
        con=con,
    )
    return buf.getvalue()


@pytest.mark.parametrize("width", [40, 80, 200])
@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("uses_runtime_api", [False, True])
def test_invoke_command_is_complete_inside_panel(width, target, uses_runtime_api):
    output = _render(width, target, uses_runtime_api=uses_runtime_api)
    rows = _panel_rows(output)
    assert "Invoke with Agent Bricks" in rows, output
    after_label = rows[rows.index("Invoke with Agent Bricks") + 1 :]
    command = agent_invoke_command(target, uses_runtime_api=uses_runtime_api)
    # Only line breaks and border padding are added: every character survives, in order.
    assert "".join("".join(after_label).split()) == "".join(command.split()), output
    assert output.splitlines()[-1].startswith("╰")


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("uses_runtime_api", [False, True])
def test_invoke_command_is_shell_copyable_from_wide_panel(shell, target, uses_runtime_api):
    executable = shutil.which(shell)
    if executable is None or (uses_runtime_api and shutil.which("uuidgen") is None):
        pytest.skip(f"requires {shell} and uuidgen for runtime examples")

    command = agent_invoke_command(target, uses_runtime_api=uses_runtime_api)
    output = _render(len(command) + 4, target, uses_runtime_api=uses_runtime_api)
    # Wide enough for the command: it is one `│ … │` line, so copying it whole drops only borders.
    lines = [line for line in output.splitlines() if "agentbricks endpoint" in line]
    assert len(lines) == 1, output
    assert lines[0].startswith("│ ") and lines[0].endswith(" │")
    copied = lines[0].removeprefix("│ ").removesuffix("│").strip()
    assert copied == command
    path = "/api/invocations" if uses_runtime_api else "/invocations"
    invocation_ids = set()
    session_ids = set()
    for _ in range(2):
        # Capture argv instead of reaching a workspace; functional tests exercise the real CLI/HTTP.
        result = subprocess.run(
            [executable, "-f", "-c", 'agentbricks() { printf "%s\\0" "$@"; }\n' + copied],
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
            session_id = uuid.UUID(body.pop("session_id"))
            assert invocation_id != session_id
            assert invocation_id not in invocation_ids
            assert session_id not in session_ids
            invocation_ids.add(invocation_id)
            session_ids.add(session_id)
        assert body == {}


@pytest.mark.parametrize("width", [40, 80, 200])
@pytest.mark.parametrize("terminal", [False, True])
def test_invoke_step_is_last_and_preserves_next_steps(width, terminal):
    buf = io.StringIO()
    con = Console(file=buf, width=width, no_color=True, force_terminal=terminal)
    render.success("Started", next_steps=_STEPS, con=con)
    without_invoke = _panel_rows(buf.getvalue())

    output = _render(width, "--url http://localhost:8000", uses_runtime_api=True, terminal=terminal)
    rows = _panel_rows(output)
    # The existing rows are untouched; the invoke rows follow them, then the bottom border.
    assert rows[: len(without_invoke)] == without_invoke
    assert rows[len(without_invoke)] == "Invoke with Agent Bricks"
    assert without_invoke[-4:] == [
        "Run locally",
        "agentbricks dev",
        "Deploy it",
        "agentbricks deploy my-agent",
    ]
    assert output.splitlines()[-1].startswith("╰")


def test_success_without_invoke_step_has_no_invoke_rows():
    buf = io.StringIO()
    con = Console(file=buf, width=80, no_color=True)
    render.success("Started", next_steps=_STEPS, con=con)
    assert "Invoke with Agent Bricks" not in buf.getvalue()
