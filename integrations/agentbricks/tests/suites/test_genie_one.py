"""`agentbricks tools add genie-one`: each framework's adapter asks Genie One and reads the rows back.

A script drives ask → poll → query result through the adapter in the project's own venv
(``genie_one_conversation.py``) instead of a model choosing the calls: a model rarely chains all
three reliably, and only the last returns the rows.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest
from harness.agentbricks_cli import AgentbricksCli
from harness.logs import log, log_command
from harness.project import Project
from harness.run import E2EError
from harness.target import TargetWorkspace

CONVERSATION_SCRIPT = pathlib.Path(__file__).with_name("genie_one_conversation.py")


# Genie One needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_genie_one(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    target_workspace: TargetWorkspace,
    framework: str,
) -> None:
    agentbricks_cli.run(
        f"init --framework {framework} --disable-chat-app {project.path}", timeout=600
    )
    agentbricks_cli.run(f"tools add genie-one --source {project.path}")
    project.pin_runtime()
    # The test venv's VIRTUAL_ENV would point uv at it instead of the project's own .venv.
    env = {
        **{name: value for name, value in os.environ.items() if name != "VIRTUAL_ENV"},
        **target_workspace.env,
    }
    _run(["uv", "sync"], project.path, env, timeout=900)
    python = project.path / ".venv" / "bin" / "python"
    result = _run(
        [str(python), str(CONVERSATION_SCRIPT), framework], project.path, env, timeout=900
    )

    transcript = json.loads(result.stdout.strip().splitlines()[-1])
    answer, rows = transcript["answer"], transcript["rows"]
    assert answer["status"] == "completed", answer
    assert answer.get("final_answer"), answer
    assert rows and rows.get("columns") and rows.get("rows"), transcript


def _run(
    argv: list[str], cwd: pathlib.Path, env: dict[str, str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    log_command(argv, cwd)
    result = subprocess.run(
        argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout, check=False
    )
    for text in (result.stdout, result.stderr):
        if text.strip():
            log(text)
    if result.returncode != 0:
        raise E2EError(
            f"{pathlib.Path(argv[0]).name} failed ({result.returncode}):\n"
            f"{result.stderr or result.stdout}"
        )
    return result
