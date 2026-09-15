"""Hermetic e2e for the local dev loop: a scaffolded agent installs, boots, and answers an invocation.

`mason init` scaffolds an agent; ``uv sync`` builds its venv; then the agent's real entrypoint
(``start-server`` — the command ``mason dev`` runs the app with) boots in a from-scratch environment
whose only Databricks target is an in-process fake model-serving endpoint, and a POST to
``/api/invocations`` must come back with the model's answer. No cloud, no auth, no workspace.

It boots the entrypoint directly rather than through ``databricks apps run-local``: run-local
resolves a real workspace at startup (config discovery + SCIM user), so it can't run in an empty
environment. Booting the entrypoint runs the same server run-local would, and stays hermetic.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import tomlkit
import yaml

_MARKER = "MASON_DEV_OK"


class _FakeServingHandler(BaseHTTPRequestHandler):
    """Minimal Databricks model-serving stand-in: config discovery + a streamed chat completion.

    ChatDatabricks calls ``/serving-endpoints/chat/completions`` with ``stream=True``; we answer with
    OpenAI-shaped SSE whose content is the marker, so the agent's reply is deterministic.
    """

    def log_message(self, format: str, *args: object) -> None:  # silence per-request logging
        pass

    def _json(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({})  # host/config discovery — an empty doc is enough for a direct boot

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": _MARKER},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_listening(port: int, proc: subprocess.Popen, timeout: float = 90) -> bool:
    """True once ``port`` accepts a connection; False if the process exits first or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _terminate(proc: subprocess.Popen) -> None:
    """Kill the server's whole process group (start-server spawns uvicorn workers)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _mason_source_under_test() -> dict:
    """Where the scaffold should install databricks-mason from — the code under test.

    `mason init` doesn't pin the scaffold to a source (it takes databricks-mason from PyPI), so the
    test supplies one, or it would validate a *released* SDK. In CI, pin the exact commit (fetchable
    because CI pushed it) so it works on any base; locally, pin the working-tree checkout (editable)
    so uncommitted changes are exercised too.
    """
    repo, sha = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_SHA")
    if repo and sha:
        return {
            "git": f"https://github.com/{repo}",
            "rev": sha,
            "subdirectory": "integrations/mason",
        }
    mason_pkg = pathlib.Path(__file__).resolve().parents[2]  # integrations/mason
    return {"path": str(mason_pkg), "editable": True}


def _pin_mason_source(pyproject: pathlib.Path, source: dict) -> None:
    """Set `[tool.uv.sources] databricks-mason = source` in the scaffold's pyproject.toml."""
    document = tomlkit.parse(pyproject.read_text())
    if "tool" not in document:
        document["tool"] = tomlkit.table()
    if "uv" not in document["tool"]:
        document["tool"]["uv"] = tomlkit.table()
    if "sources" not in document["tool"]["uv"]:
        document["tool"]["uv"]["sources"] = tomlkit.table()
    table = tomlkit.inline_table()
    table.update(source)
    document["tool"]["uv"]["sources"]["databricks-mason"] = table
    pyproject.write_text(tomlkit.dumps(document))


def test_scaffolded_agent_boots_and_answers_locally(tmp_path: pathlib.Path) -> None:
    mason = pathlib.Path(sys.executable).with_name("mason")
    uv = shutil.which("uv")
    if not mason.is_file() or uv is None:
        pytest.skip("requires the mason CLI and uv on PATH")

    # 1. Scaffold a langgraph agent (the durable runtime is on by default).
    project = tmp_path / "agent"
    subprocess.run(
        [str(mason), "init", "--framework", "langgraph", str(project)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Neutralize the profile mason init seeds into .env so the agent can't reach a real workspace.
    (project / ".env").write_text("")
    # Point the scaffold at the databricks-mason under test — `mason init` takes it from PyPI, so
    # without this the venv would build a *released* SDK, not the code being tested (CI = the commit,
    # local = the working-tree checkout).
    _pin_mason_source(project / "pyproject.toml", _mason_source_under_test())

    # 2. Build the agent's venv (installs the databricks-mason pinned above + its runtime deps).
    subprocess.run(
        [uv, "sync"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    start_server = project / ".venv" / "bin" / "start-server"
    assert start_server.is_file(), "uv sync did not install the start-server entrypoint"

    # 3. Take mason dev's OWN local manifest env, so the boot matches what `mason dev` runs the app
    #    with (the env it injects for a local run — e.g. the local-runtime marker).
    from databricks_mason.dev import _dev_entry_point

    manifest = _dev_entry_point(project / "app.yaml")
    manifest_env = {
        entry["name"]: entry["value"]
        for entry in (yaml.safe_load(manifest.read_text()).get("env") or [])
        if isinstance(entry, dict) and "name" in entry
    }
    assert manifest_env["DATABRICKS_MASON_RUNTIME_STORE_LOCAL"] == "true"
    manifest.unlink(missing_ok=True)  # don't leave the local-only manifest in the project tree

    # 4. Fake model serving.
    fake = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServingHandler)
    fake_port = fake.server_address[1]
    threading.Thread(target=fake.serve_forever, daemon=True).start()

    # 5. Boot the real entrypoint in a from-scratch environment: only PATH/HOME + project root, the
    #    fake as the sole Databricks target, and mason dev's manifest env. `env=` replaces the whole
    #    environment, so no ambient DATABRICKS_* / profile can leak in.
    #
    #    DATABRICKS_APP_NAME reproduces a real local run: `mason dev` runs the app as an Apps-style
    #    local process, where the durable runtime falls back to in-memory *only* via
    #    DATABRICKS_MASON_RUNTIME_STORE_LOCAL. The assertion above and successful boot together
    #    verify that the dev manifest selects the in-memory Runtime Store.
    app_port = _free_port()
    home = tmp_path / "home"
    home.mkdir()
    log_path = tmp_path / "server.log"
    boot_env = {
        "PATH": f"{start_server.parent}:/usr/bin:/bin",
        "HOME": str(home),
        "MASON_PROJECT_ROOT": str(project),
        "DATABRICKS_APP_NAME": "mason-dev-local",
        "DATABRICKS_HOST": f"http://127.0.0.1:{fake_port}",
        "DATABRICKS_TOKEN": "dummy",
        "PORT": str(app_port),
        **manifest_env,
    }
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [str(start_server)],
            cwd=str(project),
            env=boot_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            assert _wait_until_listening(app_port, server), (
                f"agent never started listening — it failed to boot.\n{log_path.read_text()}"
            )
            # 6. A foreground invocation must come back with the model's canned answer.
            request = urllib.request.Request(
                f"http://127.0.0.1:{app_port}/api/invocations",
                data=json.dumps(
                    {
                        "id": "00000000-0000-4000-8000-000000000000",
                        "input": [{"role": "user", "content": "hi"}],
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=40) as response:
                body = json.loads(response.read())
        finally:
            _terminate(server)
            fake.shutdown()

    assert body.get("status") == "completed", body
    assert _MARKER in json.dumps(body["output"]), body
