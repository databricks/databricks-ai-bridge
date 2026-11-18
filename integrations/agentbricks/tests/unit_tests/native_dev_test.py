"""Run the local workflow with real HTTP, SSE and child-process cleanup."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from databricks_agentbricks.clients.local_ports import free_port, port_in_use
from databricks_agentbricks.projects.agent_project import AgentProject

_SERVER = """
import os
import uvicorn
from databricks_agentkit import DurableAgentServer
from runtime.ui import install_ui
app = DurableAgentServer()
@app.invoke
async def invoke(input, context):
    await context.emit({"type": "delta", "content": input})
    return input
install_ui(app)
uvicorn.run(app, host=os.environ["HOST"], port=int(os.environ["PORT"]))
"""


@pytest.mark.skipif(os.name != "posix", reason="subprocess SIGINT cleanup requires POSIX")
@pytest.mark.parametrize("legacy", [False, True], ids=["agent-toml", "app-yaml"])
def test_dev_serves_chat_ui_and_streams_without_apps_cli(tmp_path, legacy):
    package = Path(__file__).resolve().parents[2]
    overlay = package / "src/databricks_agentbricks/templates/ui/agent-openai"
    shutil.copytree(overlay / "runtime", tmp_path / "runtime")
    shutil.copytree(overlay / "ui", tmp_path / "ui")
    (tmp_path / "server.py").write_text(_SERVER)
    command = [sys.executable, "server.py"]
    project = AgentProject.create(tmp_path, framework="openai", server="agentbricks")
    if legacy:
        (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": command}))
    else:
        project.set_dev_command(command)
    project.write()
    port = free_port()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(package / "src"), str(package.parents[1] / "src")]),
        "AGENTBRICKS_DISABLE_TELEMETRY": "1",
        "AGENTBRICKS_DISABLE_AGENT_HINT": "1",
        # No Databricks CLI, uv or uvx is available on the child PATH.
        "PATH": str(tmp_path / "empty-bin"),
        "COLUMNS": "250",
        "AGENT_MEMORY_STORE": "",
        "AGENT_SESSION_STORE": "",
    }
    base_url = f"http://127.0.0.1:{port}"
    with (tmp_path / "dev.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from databricks_agentbricks.cli import main; main()",
                "dev",
                "--source",
                str(tmp_path),
                "--no-prepare-environment",
                "--app-port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, (tmp_path / "dev.log").read_text()
                try:
                    with urllib.request.urlopen(base_url, timeout=1) as response:
                        html = response.read().decode()
                    break
                except (OSError, urllib.error.URLError):
                    assert time.monotonic() < deadline, (tmp_path / "dev.log").read_text()
                    time.sleep(0.05)
            assert "<html" in html.lower()
            request = urllib.request.Request(
                base_url + "/api/invocations",
                data=json.dumps(
                    {
                        "id": "11111111-1111-4111-8111-111111111111",
                        "input": "local hello",
                        "stream": True,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.headers["Content-Type"].startswith("text/event-stream")
                events = response.read().decode()
            assert "event: delta" in events
            assert '"content": "local hello"' in events
            assert "event: run.completed" in events
            deadline = time.monotonic() + 5
            while "Chat UI" not in (tmp_path / "dev.log").read_text():
                assert time.monotonic() < deadline, (tmp_path / "dev.log").read_text()
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    assert not port_in_use(port), "the local server survived stopping agentbricks dev"
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
