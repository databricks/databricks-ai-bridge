"""Hermetic e2e for the local dev loop: a scaffolded agent installs, boots, and answers an invocation.

`agentbricks init` scaffolds an agent; ``uv sync`` builds its venv; then the agent's real entrypoint
(``start-server`` — the command ``agentbricks dev`` runs the app with) boots in a from-scratch environment
whose only Databricks target is an in-process fake model-serving endpoint. Copying the printed
next-step command into a real shell must come back with the model's answer, for each template.
No cloud, no auth, no workspace.

It boots the entrypoint directly rather than through ``databricks apps run-local``: run-local
resolves a real workspace at startup (config discovery + SCIM user), so it can't run in an empty
environment. Booting the entrypoint runs the same server run-local would, and stays hermetic.
"""

from __future__ import annotations

import io
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
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import tomlkit
import yaml
from rich.console import Console

from databricks_agentbricks.presentation import render
from databricks_agentbricks.projects.agent_project import AgentServer

_MARKER = "AGENTBRICKS_DEV_OK"


class _FakeServingHandler(BaseHTTPRequestHandler):
    """Model-serving stand-in for Chat Completions and Responses, including streaming.

    Both protocols return the marker, so the frameworks' native HTTP clients produce a
    deterministic answer without replacing the scaffolded agent or its model client.
    """

    def log_message(self, format: str, *args: object) -> None:  # silence per-request logging
        pass

    def _json(self, obj: dict, *, status: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({})  # host/config discovery — an empty doc is enough for a direct boot

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length))
        if self.path.endswith("/responses"):
            self._responses(request)
            return
        metadata = {"id": "chatcmpl-test", "created": 1, "model": "databricks-gpt-5-2"}
        if not request.get("stream"):
            self._json(
                {
                    **metadata,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": _MARKER},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
            return
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
            chunk = {**metadata, "object": "chat.completion.chunk", **chunk}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _responses(self, request: dict) -> None:
        # Databricks requires stateless Responses calls, including native tool-loop followups.
        if any(key in request for key in ("previous_response_id", "conversation")):
            self._json(
                {"error": {"message": "Server-side Responses state is unsupported"}}, status=400
            )
            return
        text = {"type": "output_text", "text": _MARKER, "annotations": [], "logprobs": []}
        item = {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [text],
        }
        has_time_tool = any(
            tool.get("name") == "get_current_time" for tool in request.get("tools", [])
        )
        tool_outputs = [
            entry
            for entry in request.get("input", [])
            if isinstance(entry, dict) and entry.get("type") == "function_call_output"
        ]
        if has_time_tool and not tool_outputs:
            item = {
                "id": "fc-test",
                "type": "function_call",
                "call_id": "call-time-test",
                "name": "get_current_time",
                "arguments": "{}",
                "status": "completed",
            }
        response = {
            "id": "resp-tool-test" if item["type"] == "function_call" else "resp-test",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "databricks-gpt-5-2",
            "output": [item],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "metadata": {},
            "error": None,
            "incomplete_details": None,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        if not request.get("stream"):
            self._json(response)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        part = {"item_id": item["id"], "output_index": 0, "content_index": 0}
        events = [
            {
                "type": "response.created",
                "response": {**response, "status": "in_progress", "output": [], "usage": None},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    **item,
                    "status": "in_progress",
                    **({"arguments": ""} if item["type"] == "function_call" else {"content": []}),
                },
            },
        ]
        if item["type"] == "function_call":
            events.extend(
                [
                    {"type": "response.function_call_arguments.delta", **part, "delta": "{}"},
                    {
                        "type": "response.function_call_arguments.done",
                        **part,
                        "name": "get_current_time",
                        "arguments": "{}",
                    },
                ]
            )
        else:
            events.extend(
                [
                    {
                        "type": "response.content_part.added",
                        **part,
                        "part": {**text, "text": ""},
                    },
                    {
                        "type": "response.output_text.delta",
                        **part,
                        "delta": _MARKER,
                        "logprobs": [],
                    },
                    {"type": "response.output_text.done", **part, "text": _MARKER, "logprobs": []},
                    {"type": "response.content_part.done", **part, "part": text},
                ]
            )
        events.extend(
            [
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": response},
            ]
        )
        for sequence_number, event in enumerate(events):
            event["sequence_number"] = sequence_number
            self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())


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


def _pin_package_source(
    pyproject: pathlib.Path,
    package: str,
    source: dict,
    *,
    direct_requirement: str | None = None,
) -> None:
    """Pin one package source in the scaffold's ``pyproject.toml``."""
    document = tomlkit.parse(pyproject.read_text())
    if direct_requirement is not None:
        document["project"]["dependencies"].append(direct_requirement)
    if "tool" not in document:
        document["tool"] = tomlkit.table()
    if "uv" not in document["tool"]:
        document["tool"]["uv"] = tomlkit.table()
    if "sources" not in document["tool"]["uv"]:
        document["tool"]["uv"]["sources"] = tomlkit.table()
    table = tomlkit.inline_table()
    table.update(source)
    document["tool"]["uv"]["sources"][package] = table
    pyproject.write_text(tomlkit.dumps(document))


@pytest.mark.parametrize("framework", ["langgraph", "openai", "maf"])
@pytest.mark.parametrize(
    "server_kind,chat_ui", [("agentbricks", True), ("agentbricks", False), ("custom", False)]
)
def test_scaffolded_agent_boots_and_answers_locally(
    tmp_path: pathlib.Path, monkeypatch, framework: str, server_kind: str, chat_ui: bool
) -> None:
    agentbricks = pathlib.Path(sys.executable).with_name("agentbricks")
    uv = shutil.which("uv")
    if not agentbricks.is_file() or uv is None:
        pytest.skip("requires the agentbricks CLI and uv on PATH")
    shells = [path for shell in ("bash", "zsh") if (path := shutil.which(shell))]
    if not shells or (server_kind == "agentbricks" and shutil.which("uuidgen") is None):
        pytest.skip("requires bash or zsh, plus uuidgen for runtime invocation examples")

    # 1. Scaffold each supported framework/server/UI combination.
    project = tmp_path / "agent"
    init_args = [
        str(agentbricks),
        "init",
        "--framework",
        framework,
        "--server",
        server_kind,
        str(project),
    ]
    if not chat_ui:
        init_args.append("--disable-chat-app")
    subprocess.run(
        init_args,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Neutralize the profile agentbricks init seeds into .env so the agent can't reach a real workspace.
    (project / ".env").write_text("")
    if server_kind == "agentbricks":
        runtime_main = project / "runtime/main.py"
        source = runtime_main.read_text()
        source = source.replace(
            "from databricks_agentkit import DurableAgentServer",
            "from databricks_agentkit import DurableAgentServer\n"
            "from databricks_agentkit.runtime.auth import InvocationAuthPolicy",
        )
        source = source.replace(
            "app = DurableAgentServer()",
            'app = DurableAgentServer(auth_policy=InvocationAuthPolicy(("local-probe",)))',
        )
        runtime_main.write_text(source)
    # Point the scaffold at both packages under test. CI supplies their built wheels; a local run
    # falls back to the corresponding editable working-tree checkout.
    agentbricks_wheel = os.environ.get("AGENTBRICKS_WHEEL")
    agentbricks_source = (
        {"path": agentbricks_wheel}
        if agentbricks_wheel
        else {"path": str(pathlib.Path(__file__).resolve().parents[2]), "editable": True}
    )
    bridge_wheel = os.environ.get("AI_BRIDGE_WHEEL")
    bridge_source = (
        {"path": bridge_wheel}
        if bridge_wheel
        else {"path": str(pathlib.Path(__file__).resolve().parents[4]), "editable": True}
    )
    _pin_package_source(project / "pyproject.toml", "databricks-agentbricks", agentbricks_source)
    _pin_package_source(
        project / "pyproject.toml",
        "databricks-ai-bridge",
        bridge_source,
        direct_requirement="databricks-ai-bridge[memory]",
    )

    # 2. Build the agent's venv (installs the databricks-agentbricks pinned above + its runtime deps).
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

    # 3. Take agentbricks dev's OWN local manifest env, so the boot matches what `agentbricks dev` runs the app
    #    with (the env it injects for a local run — e.g. the local-runtime marker).
    from databricks_agentbricks.cli.dev import _announce_local_url, _dev_entry_point

    manifest = _dev_entry_point(project / "app.yaml")
    manifest_env = {
        entry["name"]: entry["value"]
        for entry in (yaml.safe_load(manifest.read_text()).get("env") or [])
        if isinstance(entry, dict) and "name" in entry
    }
    assert manifest_env["DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL"] == "true"
    manifest.unlink(missing_ok=True)  # don't leave the local-only manifest in the project tree

    # 4. Fake model serving.
    model_requests = []
    responses_requests = []

    class RecordingServingHandler(_FakeServingHandler):
        def do_POST(self):
            if self.path.endswith(("/responses", "/chat/completions")):
                model_requests.append(self.path)
            super().do_POST()

        def _responses(self, request):
            responses_requests.append(request)
            super()._responses(request)

    fake = ThreadingHTTPServer(("127.0.0.1", 0), RecordingServingHandler)
    fake_port = fake.server_address[1]
    threading.Thread(target=fake.serve_forever, daemon=True).start()

    # 5. Boot the real entrypoint in a from-scratch environment: only PATH/HOME + project root, the
    #    fake as the sole Databricks target, and agentbricks dev's manifest env. `env=` replaces the whole
    #    environment, so no ambient DATABRICKS_* / profile can leak in.
    #
    #    DATABRICKS_APP_NAME reproduces a real local run: `agentbricks dev` runs the app as an Apps-style
    #    local process, where the durable runtime falls back to in-memory *only* via
    #    DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL. The assertion above and successful boot together
    #    verify that the dev manifest selects the in-memory Runtime Store.
    app_port = _free_port()
    test_home = tmp_path / "home"
    test_home.mkdir()
    log_path = tmp_path / "server.log"
    boot_env = {
        "PATH": f"{start_server.parent}:/usr/bin:/bin",
        "HOME": str(test_home),
        "AGENTBRICKS_PROJECT_ROOT": str(project),
        "DATABRICKS_APP_NAME": "agent-bricks-dev-local",
        "DATABRICKS_HOST": f"http://127.0.0.1:{fake_port}",
        "DATABRICKS_TOKEN": "dummy",
        "OPENAI_AGENTS_DISABLE_TRACING": "1",
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
            # 6. Copy dev's actual output unchanged into each available shell. The line is longer
            #    than the console width: hard wraps or panel borders would break shell parsing.
            buf = io.StringIO()
            monkeypatch.setattr(render, "_stdout", Console(file=buf, width=80, no_color=True))
            _announce_local_url(project, app_port, AgentServer(server_kind))
            (tmp_path / "next-steps.txt").write_text(buf.getvalue())
            commands = [
                line
                for line in buf.getvalue().splitlines()
                if line.startswith("agentbricks endpoint")
            ]
            assert len(commands) == 1, buf.getvalue()
            cli_env = {**boot_env, "PATH": f"{agentbricks.parent}:/usr/bin:/bin"}
            invocation_ids = set()
            for shell in shells:
                for attempt in range(2):
                    result = subprocess.run(
                        [shell, "-f", "-c", commands[0]],
                        cwd=project,
                        env=cli_env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    (tmp_path / f"invoke-{pathlib.Path(shell).name}-{attempt}.log").write_text(
                        result.stdout + result.stderr
                    )
                    assert result.returncode == 0, result.stdout + result.stderr
                    body = json.loads(result.stdout)
                    assert _MARKER in json.dumps(body["output"]), body
                    if server_kind == "agentbricks":
                        assert body.get("status") == "completed", body
                        invocation_id = uuid.UUID(body["id"])
                        assert invocation_id not in invocation_ids
                        invocation_ids.add(invocation_id)
            expected_path = "/responses" if framework == "maf" else "/chat/completions"
            assert model_requests
            assert all(path.endswith(expected_path) for path in model_requests), model_requests
            if framework == "maf":
                assert all(request.get("store") is False for request in responses_requests)
            if framework == "maf" and server_kind == "agentbricks":
                tool_outputs = [
                    entry
                    for request in responses_requests
                    for entry in request.get("input", [])
                    if isinstance(entry, dict) and entry.get("type") == "function_call_output"
                ]
                assert len(tool_outputs) == 2 * len(shells), responses_requests
                for output in tool_outputs:
                    assert output["call_id"] == "call-time-test"
                    datetime.fromisoformat(output["output"])
        finally:
            _terminate(server)
            fake.shutdown()
            (tmp_path / "model-requests.json").write_text(json.dumps(model_requests, indent=2))
            if responses_requests:
                (tmp_path / "responses-requests.json").write_text(
                    json.dumps(responses_requests, indent=2)
                )
