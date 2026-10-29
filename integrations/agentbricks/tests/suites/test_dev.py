"""`agentbricks dev`: every tool added with `tools add` works on the local dev server, and every
`init` template boots and answers.

Each dev server gets its own free port, so dev runs can go in parallel.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from harness.agentbricks_cli import AgentbricksCli, free_port
from harness.fake_workspace import FAKE_MODEL_REPLY
from harness.project import Project
from harness.target import TargetWorkspace, WorkspaceKind
from tool_definitions import Tool

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
    with agentbricks_cli.start(f"dev --source {project.path} --app-port {port}") as server:
        server.wait_until_serving(port)
        for tool in tools:
            with subtests.test(tool=tool.name):
                output = agentbricks_cli.invoke(
                    f"--url http://127.0.0.1:{port} --no-auth", tool.prompt
                )
                tool.assert_called(output)


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
        output = _run_printed_invoke_command(agentbricks_cli, server.log_path)
        if target_workspace.name is WorkspaceKind.FAKE:
            assert FAKE_MODEL_REPLY in output, output


def _run_printed_invoke_command(cli: AgentbricksCli, dev_log: pathlib.Path) -> str:
    """Run the invoke command `dev` prints as its next step; returns its output.

    How the printed command quotes in each shell is `endpoint_examples_test.py`'s job.
    """
    lines = dev_log.read_text(encoding="utf-8").splitlines()
    commands = [
        line.removeprefix("│ ").removesuffix("│").strip()
        for line in lines
        if line.startswith("│ agentbricks endpoint invoke")
    ]
    assert len(commands) == 1, "\n".join(lines)
    result = cli.run_printed(commands[0], "bash")
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout).get("status") in (None, "completed"), result.stdout
    return result.stdout
