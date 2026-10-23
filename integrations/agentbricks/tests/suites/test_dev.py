"""`agentbricks dev`: every tool added with `tools add` works on the local dev server, and every
`init` template boots and answers.

Each dev server gets a free port, passed as both --app-port and PORT: the template server binds
$PORT, so an inherited PORT would override --app-port.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest
from harness.agentbricks_cli import AgentbricksCli, free_port
from harness.fake_workspace import FAKE_MODEL_REPLY
from harness.project import Project
from harness.target import TargetWorkspace, WorkspaceKind
from tool_definitions import Tool

# Local dev servers share run-local's proxy port, so all dev runs stay on one worker, in order.
pytestmark = pytest.mark.xdist_group("dev")

# The console wraps the box at its width, which would split the printed command over lines.
WIDE_CONSOLE = {"COLUMNS": "250"}

# The managed server with its chat UI on and the LangGraph framework is `test_dev`'s template.
TEMPLATES = {
    "langgraph-managed-no-ui": ["--framework", "langgraph", "--disable-chat-app"],
    "langgraph-custom": ["--framework", "langgraph", "--server", "custom"],
    "openai-managed-ui": ["--framework", "openai"],
    "openai-managed-no-ui": ["--framework", "openai", "--disable-chat-app"],
    "openai-custom": ["--framework", "openai", "--server", "custom"],
}


def test_dev(
    agentbricks_cli: AgentbricksCli, project: Project, tools: tuple[Tool, ...], subtests
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    for tool in tools:
        if tool.add_command:
            agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
        project.write_files(tool.files)

    port = free_port()
    with agentbricks_cli.start(
        f"dev --source {project.path} --app-port {port}",
        env={"PORT": str(port), **WIDE_CONSOLE},
    ) as server:
        server.wait_until_serving(port)
        for tool in tools:
            with subtests.test(tool=tool.name):
                output = agentbricks_cli.invoke(
                    f"--url http://127.0.0.1:{port} --no-auth", tool.prompt
                )
                tool.check(output)
        _check_printed_invoke_command(agentbricks_cli, server.log_path)


@pytest.mark.parametrize("template", TEMPLATES)
def test_dev_templates(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    target_workspace: TargetWorkspace,
    template: str,
) -> None:
    agentbricks_cli.run(f"init {' '.join(TEMPLATES[template])} {project.path}", timeout=600)
    project.pin_runtime()

    port = free_port()
    with agentbricks_cli.start(
        f"dev --source {project.path} --app-port {port}",
        env={"PORT": str(port), **WIDE_CONSOLE},
    ) as server:
        server.wait_until_serving(port)
        # Each template serves its own path and request shape, so the printed command is the way in.
        answer = FAKE_MODEL_REPLY if target_workspace.name is WorkspaceKind.FAKE else None
        _check_printed_invoke_command(agentbricks_cli, server.log_path, answer)


def _check_printed_invoke_command(
    cli: AgentbricksCli, dev_log: pathlib.Path, answer: str | None = None
) -> None:
    """The next step `dev` prints works when pasted into each available shell, and returns ``answer``."""
    lines = dev_log.read_text(encoding="utf-8").splitlines()
    commands = [
        line.removeprefix("│ ").removesuffix("│").strip()
        for line in lines
        if line.startswith("│ agentbricks endpoint invoke")
    ]
    assert len(commands) == 1, "\n".join(lines)
    shells = [path for name in ("bash", "zsh") if (path := shutil.which(name))]
    assert shells and shutil.which("uuidgen"), "the printed command needs a shell and uuidgen"
    for shell in shells:
        result = cli.run_printed(commands[0], shell)
        assert result.returncode == 0, result.stderr or result.stdout
        body = json.loads(result.stdout)
        assert body.get("status") in (None, "completed"), body
        if answer is not None:
            assert answer in result.stdout, result.stdout
