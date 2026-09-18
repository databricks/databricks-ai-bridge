"""Exercise MCP registration through the installed CLI against a real workspace."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MASON_MCP_E2E") != "1",
    reason="Set RUN_MASON_MCP_E2E=1 and MASON_E2E_PROFILE to run live MCP validation.",
)


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_mcp_registration_validates_before_writing(tmp_path: pathlib.Path, framework: str):
    profile = os.environ["MASON_E2E_PROFILE"]
    service = os.environ.get("MASON_E2E_MCP_SERVICE", "system.ai.web_search")
    mason = pathlib.Path(sys.executable).with_name("mason")
    assert mason.is_file(), "Install the Mason wheel into the test environment first."
    project = tmp_path / f"agent-{framework}"

    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = [str(mason), "--profile", profile, "--output", "json", *args]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        sys.stdout.write(
            f"$ {' '.join(command)}\nexit={result.returncode}\n{result.stdout}{result.stderr}\n"
        )
        if check:
            assert result.returncode == 0, result.stderr or result.stdout
        return result

    def snapshot() -> dict[str, bytes]:
        return {
            str(path.relative_to(project)): path.read_bytes()
            for path in project.rglob("*")
            if path.is_file()
        }

    run("init", "--framework", framework, str(project))
    before = snapshot()
    missing = f"{service.rsplit('.', 1)[0]}.mason_missing_{uuid.uuid4().hex}"
    rejected = run("tools", "add", "mcp", missing, "--source", str(project), check=False)
    assert rejected.returncode == 1
    error = json.loads(rejected.stderr)["error"]
    assert error["code"] in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}
    assert missing in error["message"]
    assert snapshot() == before

    added = run("tools", "add", "mcp", service, "--name", "verified", "--source", str(project))
    payload = json.loads(added.stdout)
    assert payload["changed"] is True
    assert payload["changed_files"] == [str(project / "agent.toml")]
    assert payload["tool"] == {"id": "verified", "kind": "mcp", "source": service}
    after = snapshot()
    assert after.keys() == before.keys()
    assert {name for name in before if before[name] != after[name]} == {"agent.toml"}

    duplicate = run("tools", "add", "mcp", service, "--name", "verified", "--source", str(project))
    assert json.loads(duplicate.stdout)["changed"] is False
    assert snapshot() == after
    rejected = run("tools", "add", "mcp", missing, "--source", str(project), check=False)
    assert rejected.returncode == 1
    assert snapshot() == after
    listed = run("tools", "list", "--source", str(project))
    assert json.loads(listed.stdout)["tools"] == [
        {"id": "verified", "kind": "mcp", "source": service}
    ]
    run("tools", "remove", "verified", "--source", str(project))
    listed = run("tools", "list", "--source", str(project))
    assert json.loads(listed.stdout)["tools"] == []
