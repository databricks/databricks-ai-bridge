"""Browser + real scaffold/framework/runtime evidence, with a deterministic local model API.

No live Databricks calls: the subprocess environment contains only a loopback host and dummy token.
Run with a source checkout's Python environment and --source pointing at the checkout to verify.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import click
from playwright.sync_api import sync_playwright

CHUNKS = ["# Result\n\n", "**Bold** and ", "`code`.\n\n", "- first\n", "- second\n\n", "Final."]
REQUESTS = []


class ModelService(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def respond(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if "model-services" not in self.path:
            return self.respond({})
        q = parse_qs(urlsplit(self.path).query)
        schema = q.get("parent", ["schemas/system.ai"])[0].removeprefix("schemas/")
        start = int(q.get("page_token", ["0"])[0])
        names = [f"{schema}.model-{i:02}" for i in range(25)] + [f"{schema}.z-openai-gpt"]
        services = [
            {
                "name": f"model-services/{name}",
                "supported_api_types": ["openai/v1/chat/completions"],
            }
            for name in names[start : start + 13]
        ]
        payload: dict[str, Any] = {"model_services": services}
        if start + 13 < len(names):
            payload["next_page_token"] = str(start + 13)
        self.respond(payload)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        REQUESTS.append({"path": self.path, "body": payload, "time": time.monotonic()})
        if "chat/completions" not in self.path:
            return self.respond({})
        messages = payload.get("messages", [])
        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if "CHECK_FAILURE" in last_user:
            return self.respond(
                {
                    "error": {
                        "message": "Fixture model is unavailable. Select another model.",
                        "type": "invalid_request_error",
                        "code": "MODEL_UNAVAILABLE",
                    }
                },
                400,
            )
        if "CHECK_LONG" in last_user:
            time.sleep(185)
        if "CHECK_PAUSE" in last_user:
            time.sleep(5)
        tool_test = "CHECK_TOOL" in last_user
        tool_result = (
            messages[-1].get("content") if messages and messages[-1].get("role") == "tool" else None
        )
        metadata = {
            "id": "chatcmpl-local-probe",
            "created": 1,
            "model": payload.get("model", "fixture"),
        }
        chunks = CHUNKS if not tool_test else [f"Tool completed: {tool_result}"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if tool_test and tool_result is None:
            deltas = [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_clock",
                            "type": "function",
                            "function": {"name": "get_current_time", "arguments": "{}"},
                        }
                    ],
                }
            ]
            finish = "tool_calls"
        else:
            deltas = [{"role": "assistant", "content": chunk} for chunk in chunks]
            finish = "stop"
        try:
            for delta in deltas:
                event = {
                    **metadata,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.16)
            final = {
                **metadata,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a new output directory to preserve earlier evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project = output / "agent"
    base_env = {**os.environ, "PYTHONPATH": str(source / "integrations/agentbricks/src")}
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from databricks_agentbricks.cli import main; main()",
            "init",
            "--framework",
            args.framework,
            "--server",
            "agentbricks",
            str(project),
        ],
        check=True,
        env=base_env,
        stdout=(output / "init.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    (project / ".env").write_text("")
    helper_path = source / "integrations/agentbricks/tests/functional/dev_runtime_test.py"
    spec = importlib.util.spec_from_file_location("dev_runtime_helpers", helper_path)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper._pin_package_source(
        project / "pyproject.toml",
        "databricks-agentbricks",
        {"path": str(source / "integrations/agentbricks"), "editable": True},
    )
    helper._pin_package_source(
        project / "pyproject.toml",
        "databricks-ai-bridge",
        {"path": str(source), "editable": True},
        direct_requirement="databricks-ai-bridge[memory]",
    )
    agent_py = project / "agent/agent.py"
    agent_text = agent_py.read_text().replace(
        "You are a helpful assistant.", "You are a helpful assistant. PROJECT_INSTRUCTION_MARKER"
    )
    if args.framework == "langgraph":
        agent_text = agent_text.replace(
            "        tools=tools,",
            '        system_prompt="PROJECT_INSTRUCTION_MARKER",\n        tools=tools,',
        )
    agent_py.write_text(agent_text)
    subprocess.run(
        ["uv", "sync"],
        cwd=project,
        check=True,
        stdout=(output / "sync.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    fake = ThreadingHTTPServer(("127.0.0.1", 0), ModelService)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    port = free_port()
    test_home = output / "home"
    test_home.mkdir(exist_ok=True)
    env = {
        "PATH": f"{project / '.venv/bin'}:/usr/bin:/bin",
        "DATABRICKS_CONFIG_FILE": str(test_home / "empty-config"),
        "AGENTBRICKS_PROJECT_ROOT": str(project),
        "DATABRICKS_HOST": f"http://127.0.0.1:{fake.server_port}",
        "DATABRICKS_TOKEN": "dummy",
        "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL": "true",
        "DATABRICKS_APP_NAME": "agent-bricks-local-test",
        "DATABRICKS_APP_URL": f"http://127.0.0.1:{port}",
        "OPENAI_AGENTS_DISABLE_TRACING": "1",
        "PORT": str(port),
    }
    evidence: dict[str, Any] = {
        "framework": args.framework,
        "source": str(source),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "cloud": False,
        "cases": {},
    }
    with (output / "server.log").open("w") as logs:
        server = subprocess.Popen(
            [str(project / ".venv/bin/start-server")],
            cwd=project,
            env=env,
            stdout=logs,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            assert helper._wait_until_listening(port, server), (output / "server.log").read_text()
            with sync_playwright() as pw:
                browser = pw.chromium.launch(executable_path=args.browser_executable, headless=True)
                context = browser.new_context(viewport={"width": 1440, "height": 1100})
                context.tracing.start(screenshots=True, snapshots=True, sources=True)
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{port}")
                page.wait_for_function("!document.querySelector('#prompt-input').disabled")
                page.wait_for_function("document.querySelector('#model-select').options.length > 1")
                evidence["model_options"] = page.locator("#model-select option").all_text_contents()
                evidence["mode_labels"] = page.locator(".mode-button").all_text_contents()

                if args.long_background or args.pause_background:
                    page.locator("[data-mode='background']").click()
                    page.locator("#prompt-input").fill(
                        "CHECK_LONG" if args.long_background else "CHECK_PAUSE"
                    )
                    with page.expect_response(
                        lambda r: r.url.endswith("/api/invocations") and r.request.method == "POST"
                    ) as accepted:
                        page.locator("#send-button").click()
                    invocation_id = accepted.value.json()["id"]
                    start = time.monotonic()
                    if args.long_background:
                        # Deliberate duration probe: exercise the reported real three-minute boundary.
                        page.wait_for_timeout(181000)
                        evidence["at_181_seconds"] = {
                            "status": page.locator("#run-status").inner_text(),
                            "chat": page.locator("#chat-log").inner_text(),
                        }
                        page.screenshot(path=str(output / "background-181s.png"), full_page=True)
                    else:
                        posts = []
                        page.on(
                            "request",
                            lambda request: posts.append(request.url)
                            if request.url.endswith("/api/invocations") and request.method == "POST"
                            else None,
                        )
                        page.locator("#background-wait-toggle").click()
                        page.wait_for_function(
                            "document.querySelector('#chat-log').getAttribute('aria-busy') === 'false'"
                        )
                        assert page.locator("#new-session").is_disabled()
                        page.locator("#prompt-input").fill("SHOULD_NOT_SEND")
                        page.locator("#prompt-input").press("Enter")
                        page.evaluate(
                            "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
                        )
                        assert not posts, posts
                        evidence["paused"] = {
                            "status": page.locator("#run-status").inner_text(),
                            "new_session_disabled": True,
                            "extra_invocations": len(posts),
                        }
                        page.screenshot(path=str(output / "background-paused.png"), full_page=True)
                        page.locator("#background-wait-toggle").click()
                    while time.monotonic() - start < 205:
                        state = page.request.get(
                            f"http://127.0.0.1:{port}/api/invocations/{invocation_id}"
                        ).json()
                        if state.get("status") in ("completed", "failed"):
                            evidence["server_final"] = state
                            break
                        page.wait_for_timeout(500)
                    page.wait_for_function(
                        "document.querySelector('#chat-log').getAttribute('aria-busy') === 'false'",
                        timeout=10000,
                    )
                    evidence["final_ui"] = {
                        "status": page.locator("#run-status").inner_text(),
                        "chat": page.locator("#chat-log").inner_text(),
                    }
                    if args.expect_fixed:
                        assert evidence["final_ui"]["status"] == "Ready"
                        assert evidence.get("server_final", {}).get("status") == "completed"
                        assert evidence["final_ui"]["chat"].count("Final.") == 1
                        assert not errors, errors
                        if args.long_background:
                            assert evidence["at_181_seconds"]["status"] != "Error"
                    if args.pause_background:
                        assert evidence["final_ui"]["status"] == "Ready"
                        assert evidence["final_ui"]["chat"].count("Final.") == 1
                        assert not posts
                    page.screenshot(path=str(output / "background-final.png"), full_page=True)
                    context.tracing.stop(path=str(output / "browser-trace.zip"))
                    browser.close()
                    click.echo(
                        json.dumps(
                            {
                                "at_181_seconds": evidence.get("at_181_seconds"),
                                "paused": evidence.get("paused"),
                                "final_ui": evidence["final_ui"],
                                "server_status": evidence.get("server_final", {}).get("status"),
                            },
                            indent=2,
                        )
                    )
                    return

                def send(text, mode):
                    assistant_count = page.locator(".message.assistant").count()
                    page.locator(f"[data-mode='{mode}']").click()
                    page.locator("#prompt-input").fill(text)
                    page.locator("#send-button").click()
                    page.wait_for_function(
                        "document.querySelector('#chat-log').getAttribute('aria-busy') === 'false'",
                        timeout=45000,
                    )
                    return {
                        "status": page.locator("#run-status").inner_text(),
                        "text": page.locator("#chat-log").inner_text(),
                        "new_answers": page.locator(
                            ".message.assistant .message-content"
                        ).all_text_contents()[assistant_count:],
                    }

                page.evaluate("""() => {
                  window.probeMutations = [];
                  new MutationObserver(() => {
                    const text = document.querySelector('#chat-log').innerText;
                    if (text !== window.probeMutations.at(-1)?.text) window.probeMutations.push({at: performance.now(), text});
                  }).observe(document.querySelector('#chat-log'), {subtree: true, childList: true, characterData: true});
                }""")
                evidence["cases"]["streaming"] = send("CHECK_MARKDOWN", "streaming")
                evidence["cases"]["streaming"]["mutations"] = page.evaluate("window.probeMutations")
                evidence["cases"]["streaming"]["rendered_bold"] = (
                    page.locator("#chat-log strong").filter(has_text="Bold").count()
                )
                evidence["cases"]["streaming"]["rendered_code"] = (
                    page.locator("#chat-log code").filter(has_text="code").count()
                )
                page.screenshot(path=str(output / "streaming.png"), full_page=True)
                evidence["cases"]["sync"] = send("CHECK_SYNC", "sync")
                evidence["cases"]["background"] = send("CHECK_BACKGROUND", "background")
                evidence["cases"]["stream_failure"] = send("CHECK_FAILURE", "streaming")
                page.screenshot(path=str(output / "failure.png"), full_page=True)
                evidence["cases"]["background_failure"] = send("CHECK_FAILURE", "background")
                options = page.locator("#model-select option").evaluate_all(
                    "els => els.map(e => e.value)"
                )
                override = next((o for o in options if "model-01" in o), None)
                if override:
                    page.locator("#model-select").select_option(override)
                    evidence["cases"]["override_tool"] = send("CHECK_TOOL", "streaming")
                    calls = [
                        r["body"]
                        for r in REQUESTS
                        if "chat/completions" in r["path"] and r["body"].get("model") == override
                    ]
                    evidence["cases"]["override_tool"]["model"] = override
                    evidence["cases"]["override_tool"]["requests"] = calls
                evidence["console_errors"] = errors
                if args.expect_fixed:
                    assert len(evidence["model_options"]) == 28, evidence["model_options"]
                    assert evidence["cases"]["streaming"]["rendered_bold"] == 1
                    assert evidence["cases"]["streaming"]["rendered_code"] == 1
                    assert evidence["cases"]["stream_failure"]["status"] == "Error"
                    assert evidence["cases"]["background_failure"]["status"] == "Error"
                    assert all(
                        evidence["cases"][mode]["status"] == "Ready"
                        for mode in ("streaming", "sync", "background", "override_tool")
                    )
                    for mode in ("streaming", "sync", "background"):
                        answers = evidence["cases"][mode]["new_answers"]
                        assert len(answers) == 1 and answers[0].count("Final.") == 1, answers
                    answers = evidence["cases"]["override_tool"]["new_answers"]
                    assert len(answers) == 1 and "Tool completed:" in answers[0], answers
                    assert not errors, errors
                    calls = evidence["cases"]["override_tool"]["requests"]
                    assert any(
                        "PROJECT_INSTRUCTION_MARKER" in str(call["messages"]) for call in calls
                    )
                    assert any(
                        message.get("role") == "tool"
                        for call in calls
                        for message in call["messages"]
                    )
                    updates = evidence["cases"]["streaming"]["mutations"]
                    assert len(updates) >= 6, updates
                    # Verify the pacing correction rather than a single environment-specific latency.
                    assert updates[-1]["at"] - updates[1]["at"] >= 500, updates
                context.tracing.stop(path=str(output / "browser-trace.zip"))
                browser.close()
        finally:
            helper._terminate(server)
            fake.shutdown()
            evidence["model_requests"] = REQUESTS
            (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    click.echo(
        json.dumps(
            {
                "framework": args.framework,
                "output": str(output),
                "model_count": len(evidence.get("model_options", [])),
                "cases": {
                    k: {
                        a: b
                        for a, b in v.items()
                        if a in ("status", "rendered_bold", "rendered_code")
                    }
                    for k, v in evidence["cases"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--framework", choices=["openai", "langgraph"], required=True)
    parser.add_argument("--long-background", action="store_true")
    parser.add_argument("--pause-background", action="store_true")
    parser.add_argument(
        "--expect-fixed",
        action="store_true",
        help="Assert the corrected behavior; omit when recording the before-fix baseline.",
    )
    parser.add_argument(
        "--browser-executable",
        default=shutil.which("google-chrome"),
        help="Browser path, or use Playwright's installed Chromium.",
    )
    run(parser.parse_args())
