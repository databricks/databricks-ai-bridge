"""Offline Chromium checks for the sandbox UI, using explicit fake HTTP contracts.

Run with a Python environment containing pytest and playwright. These tests do not
exercise Lakebase, Databricks ingress, or real sandbox startup/stop behavior.
"""

import copy
import functools
import http.server
import json
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

import databricks_agentkit.sandbox as sandbox_package

playwright = pytest.importorskip("playwright.sync_api")
STATIC = Path(sandbox_package.__file__).parent / "static"


class Runtime:
    def __init__(self):
        self.agent = {
            "id": "agent-one",
            "name": "Support agent",
            "source_kind": "workspace",
            "source_path": "/Workspace/Users/demo/A #?% 日本",
            "source_hash": "a" * 64,
            "entrypoint": "agent:run",
            "file_count": 3,
            "git_source": {
                "commit": "b" * 40,
                "requested_revision": "feature/chat",
                "subdirectory": "agent",
            },
        }
        self.config = {
            "workspace": "https://example.cloud.databricks.com",
            "sandbox_scope": "session",
            "default_environment": "support",
            "default_agent_id": self.agent["id"],
            "environment_mode": "on-demand",
            "service_principal": "app-principal",
        }
        self.agents = [self.agent]
        self.runs = []
        self.preparations = {}
        self.preparation_posts = []
        self.posts = []
        self.sandbox_reads = []
        self.stops = []
        self.status = {}
        self.ambiguous = False
        self.history_error = False
        self.events = []
        self.event_stream = False
        self.stop_error = False
        self.models = ["system.ai.claude-sonnet-5-5", "system.ai.gpt-5-mini"]
        self.models_error = False

    def run(self, session, *, run_id="run-one", status="running", stage="Running agent", text=""):
        row = {
            "id": run_id,
            "agent_id": self.agent["id"],
            "session_id": session,
            "invocation_id": f"invocation-{run_id}",
            "input": {"message": f"Hello {session}"},
            "status": status,
            "stage": stage,
            "created_at": "2026-11-09T12:00:00Z",
            "stream": {"text": text},
            "result": None,
            "error": None,
        }
        self.runs.append(row)
        return row

    def complete(self, row, reply="**Finished**"):
        row.update(
            status="succeeded",
            stage="Complete",
            result={
                "sandbox_id": f"sandbox-{row['session_id']}",
                "environment_reused": False,
                "timings_ms": {"environment_startup": 2345, "total": 3200},
                "result": {"reply": reply},
            },
        )
        self.status[row["session_id"]] = "running"

    def route(self, route):
        request = route.request
        parts = urlsplit(request.url)
        path, query = parts.path, parse_qs(parts.query)
        body = request.post_data_json if request.method == "POST" else None
        data, status = None, 200
        if path == "/api/config":
            data = self.config
        elif path == "/api/models":
            data, status = (
                ({"detail": "Discovery unavailable"}, 503)
                if self.models_error
                else ({"available": self.models}, 200)
            )
        elif path == "/api/agents" and body is None:
            data = self.agents
        elif path == "/api/agents" and body:
            data = {
                **self.agent,
                "id": "new-agent",
                "name": body["name"],
                "source_kind": body["source_kind"],
                "source_path": body["source_path"],
            }
            self.agents.append(data)
        elif path.endswith("/sandbox"):
            session = query["session_id"][0]
            self.sandbox_reads.append(session)
            value = self.status.get(session, "not_created")
            data = {
                "session_id": session,
                "agent_id": path.split("/")[3],
                "sandbox_scope": self.config["sandbox_scope"],
                "sandbox_id": None if value == "not_created" else f"sandbox-{session}",
                "status": value,
                "execution_principal": "app-principal",
                "source_hash": "a" * 64,
                "initial_bootstrap_timings_ms": {"install": 1234},
                "preparation": self.preparations.get(session),
            }
        elif path.endswith("/sessions") and body:
            self.preparation_posts.append(body)
            session = body["session_id"]
            data = self.preparations[session] = {
                "agent_id": self.agent["id"],
                "session_id": session,
                "status": "queued",
                "stage": "Waiting to prepare session",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            status = 202
        elif path.startswith("/api/agents/") and "/sessions/" in path and body is None:
            data = self.preparations.get(path.rsplit("/", 1)[-1], {"status": "not_requested"})
        elif path.endswith("/sessions"):
            data = [
                {"session_id": row["session_id"], "status": row["status"]}
                for row in self.runs
                if row["agent_id"] == path.split("/")[3]
            ]
        elif path == "/api/runs" and body is None:
            if self.history_error:
                data, status = {"detail": "History unavailable"}, 503
            else:
                data = [
                    row
                    for row in self.runs
                    if row["session_id"] == query["session_id"][0]
                    and row["agent_id"] == query["agent_id"][0]
                ]
        elif path in {"/api/runs", "/api/environments/support/runs"} and body:
            self.posts.append(copy.deepcopy(body))
            data = next(
                (
                    row
                    for row in self.runs
                    if row["session_id"] == body["session_id"]
                    and row["invocation_id"] == body["invocation_id"]
                ),
                None,
            )
            if data is None:
                data = self.run(
                    body["session_id"],
                    run_id=f"run-{len(self.runs)}",
                    status="queued",
                    stage="Queued",
                )
                data.update(invocation_id=body["invocation_id"], input=body["input"])
            elif body.get("retry") and data["status"] in {"failed", "interrupted"}:
                data.update(status="queued", stage="Queued", error=None)
            if self.ambiguous:
                self.ambiguous = False
                route.abort("failed")
                return
        elif path.endswith("/events"):
            if not self.event_stream:
                route.abort("failed")
            else:
                route.fulfill(
                    status=200,
                    content_type="text/event-stream",
                    body="".join(f"event: run\ndata: {json.dumps(row)}\n\n" for row in self.events),
                )
            return
        elif path.startswith("/api/runs/"):
            data = next(row for row in self.runs if row["id"] == path.split("/")[3])
        elif path == "/sessions/stop":
            self.stops.append(body)
            if self.stop_error:
                data, status = {"detail": "Shutdown timed out"}, 504
            else:
                self.status[body["session_id"]] = "stopped"
                data = {
                    "status": "stopped",
                    "sandbox_id": f"sandbox-{body['session_id']}",
                    "timings_ms": {"total": 1500},
                }
        else:
            raise AssertionError(f"Unexpected mocked API: {request.method} {path}")
        route.fulfill(status=status, content_type="application/json", body=json.dumps(data))


@pytest.fixture(scope="module")
def origin():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    handler = functools.partial(Handler, directory=str(STATIC.parent))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join()


@pytest.fixture(scope="module")
def browser():
    executable = (
        os.getenv("SANDBOX_UI_CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
    )
    if not executable:
        pytest.skip("Set SANDBOX_UI_CHROME to a Chromium executable")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        yield browser
        browser.close()


@pytest.fixture
def ui(browser, origin):
    runtime = Runtime()
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/api/**", runtime.route)
    page.route("**/sessions/stop", runtime.route)

    def open_page(*, tab="sandbox"):
        page.goto(f"{origin}/static/index.html")
        page.wait_for_function("state.context && !state.loading")
        if tab:
            page.locator(f"#{tab}-tab").click()
        return page

    yield runtime, page, open_page
    assert not errors, f"Browser exceptions: {errors}"
    context.close()


def send(page, text="Hello"):
    page.locator("#prompt-input").fill(text)
    page.locator("#send-button").click()
    playwright.expect(page.locator("#chat-log .message.user")).to_have_count(1)


def test_model_selection_reaches_invocation_and_default_omits_override(ui):
    runtime, page, open_page = ui
    open_page()
    select = page.get_by_role("combobox", name="Agent model")
    playwright.expect(select).to_be_enabled()
    select.select_option("system.ai.gpt-5-mini")
    send(page)
    page.wait_for_function("state.runs.length === 1")
    assert runtime.posts[0]["input"] == {"message": "Hello", "model": "system.ai.gpt-5-mini"}
    playwright.expect(select).to_be_disabled()
    runtime.complete(runtime.runs[0])
    playwright.expect(select).to_be_enabled(timeout=6000)
    page.locator("#new-session").click()
    playwright.expect(select).to_have_value("system.ai.gpt-5-mini")
    playwright.expect(select).to_be_enabled()
    select.select_option("")
    send(page)
    page.wait_for_function("state.runs.length === 1")
    assert runtime.posts[-1]["input"] == {"message": "Hello"}


def test_model_choice_survives_uncertain_submission_reload_and_original_retry(ui):
    runtime, page, open_page = ui
    runtime.ambiguous = True
    open_page()
    select = page.get_by_role("combobox", name="Agent model")
    playwright.expect(select).to_be_enabled()
    select.select_option("system.ai.gpt-5-mini")
    send(page)
    playwright.expect(page.locator("#retry-run")).to_have_text("Reconnect this invocation")
    playwright.expect(select).to_be_disabled()
    page.reload()
    playwright.expect(page.locator("#chat-log .message.user")).to_have_count(1)
    assert len(runtime.posts) == 1
    row = runtime.runs[0]
    row.update(status="failed", error="Rate limit", stage="Failed")
    playwright.expect(page.locator("#retry-run")).to_have_text(
        "Retry this invocation", timeout=6000
    )
    playwright.expect(select).to_be_enabled()
    select.select_option("system.ai.claude-sonnet-5-5")
    page.locator("#retry-run").click()
    page.wait_for_function("state.runs.length === 2 && state.runs[1].status === 'queued'")
    assert runtime.posts[1]["input"] == runtime.posts[0]["input"]
    assert runtime.posts[1]["input"]["model"] == "system.ai.gpt-5-mini"
    assert runtime.posts[1]["invocation_id"] != runtime.posts[0]["invocation_id"]


def test_model_discovery_failure_keeps_chat_available_and_can_reload_models(ui):
    runtime, page, open_page = ui
    runtime.models_error = True
    open_page()
    playwright.expect(page.locator("#model-notice")).to_be_visible()
    select = page.get_by_role("combobox", name="Agent model")
    playwright.expect(select).to_have_value("")
    playwright.expect(select).to_be_disabled()
    page.locator("#prompt-input").fill("Hello")
    playwright.expect(page.locator("#send-button")).to_be_enabled()
    runtime.models_error = False
    page.locator("#reload-models").click()
    playwright.expect(select).to_be_enabled()
    playwright.expect(page.locator("#model-notice")).to_be_hidden()
    assert runtime.posts == []


def test_model_selection_is_page_local_and_fits_mobile(ui):
    runtime, page, open_page = ui
    runtime.models.append("system.ai.a-model-with-a-very-long-name-that-must-fit-the-composer")
    open_page()
    select = page.get_by_role("combobox", name="Agent model")
    playwright.expect(select).to_be_enabled()
    select.select_option(runtime.models[-1])
    page.set_viewport_size({"width": 375, "height": 812})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    composer = page.locator("#composer").bounding_box()
    picker = select.bounding_box()
    assert picker["x"] >= composer["x"]
    assert picker["x"] + picker["width"] <= composer["x"] + composer["width"]
    assert page.locator("#composer-hint").bounding_box()["height"] < 50
    assert page.locator(".composer-footer").bounding_box()["height"] < 100
    page.reload()
    playwright.expect(select).to_be_enabled()
    playwright.expect(select).to_have_value("")


def lifecycle(**steps):
    return {
        "steps": {
            key: steps.get(key, {"status": "pending"})
            for key in ("sandbox_start", "source_copy", "install_build", "agent_run")
        }
    }


def test_events_default_tabs_keyboard_and_mobile(ui):
    runtime, page, open_page = ui
    open_page(tab=None)
    playwright.expect(page.locator("#events-tab")).to_have_attribute("aria-selected", "true")
    playwright.expect(page.locator("#events-panel")).to_be_visible()
    playwright.expect(page.locator("#sandbox-panel")).to_be_hidden()
    playwright.expect(page.locator("#event-log")).to_contain_text("No events in this session")
    page.locator("#events-tab").focus()
    page.keyboard.press("ArrowRight")
    playwright.expect(page.locator("#sandbox-tab")).to_be_focused()
    playwright.expect(page.locator("#sandbox-tab")).to_have_attribute("aria-selected", "true")
    playwright.expect(page.locator("#sandbox-panel")).to_be_visible()
    playwright.expect(page.locator("#stop-session")).to_have_count(1)
    playwright.expect(page.locator("#view-session-source")).to_have_count(1)
    page.keyboard.press("Home")
    playwright.expect(page.locator("#events-tab")).to_be_focused()
    page.keyboard.press("End")
    playwright.expect(page.locator("#sandbox-tab")).to_be_focused()
    page.set_viewport_size({"width": 375, "height": 812})
    playwright.expect(page.locator(".right-rail")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.locator("#events-tab").click()
    assert page.locator("#events-panel").bounding_box()["width"] <= 375
    assert runtime.posts == runtime.stops == []


@pytest.mark.parametrize(
    "mode,label",
    [
        ("managed", "Managed Session Store"),
        ("sandbox-sqlite", "Sandbox SQLite"),
        ("custom", "Custom handler"),
    ],
)
def test_workbench_capabilities_report_actual_configuration(ui, mode, label):
    runtime, page, open_page = ui
    runtime.config["workbench"] = {
        "framework": "langgraph",
        "sessions": {"mode": mode, "store": "catalog.schema.sessions"},
        "tools": ["calculator", "<img src=x onerror=window.pwned=1>"],
        "tracing": {
            "configured": True,
            "experiment_name": "Sandbox experiment",
            "url": "https://example.com/experiments/123",
        },
    }
    open_page(tab=None)
    playwright.expect(page.locator("#session-mode-value")).to_have_text(label)
    playwright.expect(page.locator("#tools-mode-value")).to_have_text("2 declared bindings")
    playwright.expect(page.locator("#tracing-mode-value")).to_have_text("Configured")
    page.locator('[data-capability="tools"] button').click()
    playwright.expect(page.locator("#tools-detail")).to_contain_text("<img src=x")
    playwright.expect(page.locator("#tools-detail img")).to_have_count(0)
    page.locator('[data-capability="tracing"] button').click()
    playwright.expect(page.locator("#tracing-detail")).to_contain_text("Sandbox experiment")
    assert page.locator("#tracing-detail a").get_attribute("href").endswith("/123")
    assert page.locator("#tracing-detail a").get_attribute("rel") == "noopener noreferrer"
    assert page.locator(".capability-copy strong").all_text_contents() == [
        "Streaming",
        "Sessions",
        "Tools",
        "Traces",
    ]
    assert page.evaluate("window.pwned") is None


def test_old_config_does_not_claim_managed_tools_or_tracing(ui):
    _, page, open_page = ui
    open_page(tab=None)
    playwright.expect(page.locator("#session-mode-value")).to_have_text("Run history")
    playwright.expect(page.locator("#tools-mode-value")).to_have_text("Not reported")
    playwright.expect(page.locator("#tracing-mode-value")).to_have_text("Not reported")
    page.locator('[data-capability="session"] button').click()
    playwright.expect(page.locator("#session-detail")).to_contain_text("storage was not reported")
    playwright.expect(page.locator("#tools-status")).not_to_have_class("status-orb enabled")


def test_events_actual_steps_final_tools_and_snapshot_deduplication(ui):
    runtime, page, open_page = ui
    row = runtime.run("events-session", stage="Creating sandbox")
    row["lifecycle"] = lifecycle(
        sandbox_start={"status": "running", "started_at": "2026-11-09T12:00:00Z"}
    )
    page.add_init_script("""window.sources = []; window.EventSource = class extends EventTarget {
      constructor() { super(); window.sources.push(this); } close() {} };
      localStorage.setItem('sandbox-chat:session:agent-one', 'events-session');""")
    open_page(tab=None)
    playwright.expect(page.locator("#sandbox-tab-badge")).to_have_text("Running")
    playwright.expect(page.locator('#event-log [data-category="infrastructure"]')).to_have_count(2)
    playwright.expect(page.locator('#event-log [data-category="tools"]')).to_have_count(0)
    runtime.complete(row, "A traced answer")
    row["result"]["lifecycle"] = lifecycle(
        sandbox_start={"status": "completed", "duration_ms": 100},
        source_copy={"status": "reused"},
        install_build={"status": "reused"},
        agent_run={"status": "completed", "duration_ms": 400},
    )
    tool = {
        "call_id": "call-one",
        "name": "calculator",
        "status": "completed",
        "timestamp": "2026-11-09T12:00:01Z",
    }
    row["result"]["result"].update(
        tool_calls=["fallback-must-not-duplicate"],
        tool_events=[tool, tool, {**tool, "call_id": "call-two", "status": "failed"}],
        trace={"trace_id": "trace-one", "url": "https://example.com/traces/trace-one"},
    )
    page.evaluate(
        "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))",
        row,
    )
    playwright.expect(page.locator("#event-log .event-entry")).to_have_count(8)
    page.evaluate("row => { mergeRun(row); mergeRun(row); }", row)
    playwright.expect(page.locator("#event-log .event-entry")).to_have_count(8)
    playwright.expect(page.locator("#sandbox-tab-badge")).to_be_hidden()
    playwright.expect(page.locator("#chat-log .message-trace")).to_have_count(1)
    assert page.locator("#chat-log .message-trace").get_attribute("href").endswith("/trace-one")
    page.locator('.event-filter[data-filter="tools"]').click()
    playwright.expect(page.locator("#event-log .event-entry")).to_have_count(2)
    playwright.expect(page.locator('#event-log [data-status="failed"]')).to_have_count(1)
    playwright.expect(page.locator("#event-log")).to_contain_text("calculator · failed")
    assert "fallback-must-not-duplicate" not in page.locator("#event-log").inner_text()
    assert "started" not in page.locator("#event-log").inner_text()
    page.locator('.event-filter[data-filter="infrastructure"]').click()
    playwright.expect(page.locator("#event-log .event-entry")).to_have_count(3)
    page.locator('.event-filter[data-filter="traces"]').click()
    playwright.expect(page.locator("#event-log .event-entry")).to_have_count(1)
    assert page.locator("#event-log a").get_attribute("rel") == "noopener noreferrer"


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "javascript:window.pwned=1",
        "data:text/html,<script>alert(1)</script>",
        "https://user:secret@example.com/trace",
    ],
)
def test_trace_links_and_tool_names_are_safe_and_bound_to_answers(ui, unsafe_url):
    runtime, page, open_page = ui
    runtime.config["workbench"] = {"tracing": {"configured": True, "url": unsafe_url}}
    for index, url in enumerate(("https://example.com/trace/one", unsafe_url)):
        row = runtime.run("safe-events", run_id=f"trace-run-{index}")
        runtime.complete(row, f"Answer {index}")
        row["result"]["result"].update(
            tool_calls=["<img src=x onerror=window.pwned=1>"],
            trace={"trace_id": f"trace-{index}", "url": url},
        )
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'safe-events')")
    open_page(tab=None)
    playwright.expect(page.locator('[data-run-id="trace-run-0"] .message-trace')).to_have_count(1)
    playwright.expect(page.locator('[data-run-id="trace-run-1"] .message-trace')).to_have_count(0)
    playwright.expect(page.locator("#event-log a")).to_have_count(1)
    playwright.expect(page.locator("#event-log img")).to_have_count(0)
    playwright.expect(page.locator("#event-log")).to_contain_text("<img src=x")
    playwright.expect(page.locator("#event-log")).to_contain_text("Trace link unavailable")
    page.locator('[data-capability="tracing"] button').click()
    playwright.expect(page.locator("#tracing-detail a")).to_have_count(0)
    assert page.evaluate("window.pwned") is None


def test_events_session_switch_ignores_stale_snapshots_and_new_chat_is_empty(ui):
    runtime, page, open_page = ui
    old = runtime.run("event-old", stage="Installing dependencies")
    new = runtime.run("event-new", run_id="new-run")
    runtime.complete(new, "Other answer")
    new["result"]["result"]["tool_calls"] = ["other-session-tool"]
    page.add_init_script("""window.sources = []; window.EventSource = class extends EventTarget {
      constructor() { super(); window.sources.push(this); } close() {} };
      localStorage.setItem('sandbox-chat:session:agent-one', 'event-old');""")
    open_page(tab=None)
    page.locator(".session-open-button", has_text="event-new").click()
    playwright.expect(page.locator("#event-log")).to_contain_text("other-session-tool")
    runtime.complete(old, "Late answer")
    old["result"]["result"]["tool_calls"] = ["stale-tool-must-not-appear"]
    page.evaluate(
        "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))",
        old,
    )
    assert "stale-tool" not in page.locator("#event-log").inner_text()
    assert "Late answer" not in page.locator("#chat-log").inner_text()
    page.locator("#new-session").click()
    playwright.expect(page.locator("#event-log")).to_contain_text("No events in this session")
    playwright.expect(page.locator("#event-count")).to_have_text("0 events")
    playwright.expect(page.locator("#sandbox-tab-badge")).to_be_hidden()
    assert runtime.posts == []


def test_empty_tool_events_does_not_claim_requested_tools_completed(ui):
    runtime, page, open_page = ui
    row = runtime.run("no-completions")
    runtime.complete(row, "Approval is required")
    row["result"]["result"].update(tool_calls=["requested-only"], tool_events=[])
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'no-completions')")
    open_page(tab=None)
    playwright.expect(page.locator('#event-log [data-category="tools"]')).to_have_count(0)
    assert "requested-only" not in page.locator("#event-log").inner_text()


def test_current_turn_live_steps_first_text_and_nested_details(ui):
    runtime, page, open_page = ui
    row = runtime.run("timed-session", stage="Creating sandbox")
    started = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    row["lifecycle"] = lifecycle(sandbox_start={"status": "running", "started_at": started})
    page.add_init_script("""window.sources = []; window.EventSource = class extends EventTarget {
      constructor() { super(); window.sources.push(this); } close() {} };
      localStorage.setItem('sandbox-chat:session:agent-one', 'timed-session');""")
    open_page()
    playwright.expect(page.locator("#sandbox-start-duration")).to_contain_text("Running ·")
    playwright.expect(page.locator("#sandbox-start-duration")).to_contain_text("elapsed")
    playwright.expect(page.locator("#source-copy-duration")).to_have_text("Pending")
    playwright.expect(page.locator("#install-build-duration")).to_have_text("Pending")
    playwright.expect(page.locator("#agent-run-duration")).to_have_text("Pending")
    playwright.expect(page.locator("#first-text-duration")).to_have_text("Waiting")
    page.wait_for_function("window.sources.length === 1")
    dispatch = "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))"
    row["lifecycle"] = lifecycle(
        sandbox_start={"status": "completed", "duration_ms": 1100},
        source_copy={"status": "completed", "duration_ms": 1200},
        install_build={
            "status": "completed",
            "duration_ms": 2500,
            "details_ms": {"dependency_install": 900, "project_build": 400},
        },
        agent_run={"status": "running", "started_at": started},
    )
    row["lifecycle"]["preparation_ms"] = 4800
    row["stage"] = "Running agent"
    page.evaluate(dispatch, row)
    playwright.expect(page.locator("#startup-duration")).to_have_text("4.8 s")
    playwright.expect(page.locator("#sandbox-start-duration")).to_have_text("1.1 s")
    playwright.expect(page.locator("#source-copy-duration")).to_have_text("1.2 s")
    playwright.expect(page.locator("#install-build-duration")).to_have_text("2.5 s")
    playwright.expect(page.locator("#agent-run-duration")).to_contain_text("Running ·")
    row["stream"] = {"text": "First response", "first_token_ms": 9234}
    page.evaluate(dispatch, row)
    playwright.expect(page.locator("#first-text-duration")).to_have_text("9.2 s")
    playwright.expect(page.locator("#first-text-detail")).to_have_text(
        "Processing started → text observed by App. Excludes queue time and browser delivery."
    )
    assert len(runtime.sandbox_reads) == 1
    runtime.complete(row)
    row["lifecycle"]["steps"]["agent_run"] = {
        "status": "completed",
        "duration_ms": 14400,
        "details_ms": {"prepare": 8000, "agent_execution": 5000},
    }
    page.evaluate(dispatch, row)
    playwright.expect(page.locator("#agent-run-duration")).to_have_text("14.4 s")
    playwright.expect(page.locator("#startup-duration")).to_have_text("4.8 s")
    page.locator("#install-build-details summary").click()
    page.locator("#agent-run-details summary").click()
    playwright.expect(page.locator("#install-build-metrics")).to_contain_text("Dependencies")
    playwright.expect(page.locator("#install-build-metrics")).to_contain_text("0.9 s")
    playwright.expect(page.locator("#agent-run-metrics")).to_contain_text("Handler preparation")
    playwright.expect(page.locator("#agent-run-metrics")).to_contain_text("8.0 s")
    playwright.expect(page.locator("#timing-detail")).to_contain_text(
        "included in their step totals"
    )
    page.set_viewport_size({"width": 375, "height": 812})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert page.locator("#lifecycle-timings").bounding_box()["width"] <= 375


def test_worker_preparation_substeps_use_completed_numbers_and_keep_legacy_aggregates(ui):
    runtime, page, open_page = ui
    row = runtime.run("preparation-details")
    page.add_init_script("""window.sources = []; window.EventSource = class extends EventTarget {
      constructor() { super(); window.sources.push(this); } close() {} };
      localStorage.setItem('sandbox-chat:session:agent-one', 'preparation-details');""")
    open_page()
    playwright.expect(page.locator("#agent-run-details")).to_be_hidden()
    playwright.expect(page.locator("#agent-run-metrics")).to_be_empty()
    timings = {
        "framework_import": 1200,
        "project_import": 2300,
        "bindings": 300,
        "configuration": 500,
        "session_open": 0,
        "agent_build": 800,
        "mcp_discovery": 500,
        "prepare": 5600,
        "agent_execution": 2400,
    }
    runtime.complete(row)
    row["result"]["result"]["worker_timings_ms"] = timings
    row["result"]["lifecycle"] = lifecycle(
        agent_run={"status": "completed", "duration_ms": 9000, "details_ms": timings}
    )
    page.evaluate(
        "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))",
        row,
    )
    page.locator("#agent-run-details summary").click()
    assert page.locator("#agent-run-metrics dt").all_text_contents() == [
        "Framework imports",
        "Agent project imports",
        "Source bindings",
        "Agent configuration",
        "Session setup",
        "Agent construction",
        "MCP tool discovery",
        "Handler preparation",
        "Agent execution",
    ]
    assert page.locator("#agent-run-metrics dd").all_text_contents() == [
        "1.2 s",
        "2.3 s",
        "0.3 s",
        "0.5 s",
        "0.0 s",
        "0.8 s",
        "0.5 s",
        "5.6 s",
        "2.4 s",
    ]
    playwright.expect(page.locator("#agent-run-duration")).to_have_text("9.0 s")
    playwright.expect(page.locator("#timing-detail")).to_contain_text(
        "included in their step totals"
    )
    row["result"].pop("lifecycle")
    row["result"]["timings_ms"]["worker_process_and_execution"] = 9000
    row["result"]["result"]["worker_timings_ms"] = {"prepare": 5600, "agent_execution": 2400}
    page.evaluate("row => mergeRun(row)", row)
    assert page.locator("#agent-run-metrics dt").all_text_contents() == [
        "Handler preparation",
        "Agent execution",
    ]
    assert page.locator("#agent-run-metrics dd").all_text_contents() == ["5.6 s", "2.4 s"]
    playwright.expect(page.locator("#agent-run-duration")).to_have_text("9.0 s")
    assert runtime.posts == []


def test_reused_installation_ignores_cached_metrics_and_sync_lifecycle_reloads(ui):
    runtime, page, open_page = ui
    row = runtime.run("warm-session")
    runtime.complete(row)
    row["result"].update(
        environment_reused=True,
        initial_bootstrap_timings_ms={"dependency_install": 999999},
        lifecycle=lifecycle(
            sandbox_start={"status": "completed", "duration_ms": 0},
            source_copy={"status": "reused", "duration_ms": 999999},
            install_build={
                "status": "reused",
                "details_ms": {"dependency_install": 999999},
            },
            agent_run={"status": "completed", "duration_ms": 9000},
        ),
    )
    row["result"]["lifecycle"]["preparation_ms"] = 630
    row["stream"]["first_token_ms"] = 0
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'warm-session')")
    open_page()
    for reload in (False, True):
        if reload:
            page.reload()
            page.locator("#sandbox-tab").click()
        playwright.expect(page.locator("#startup-duration")).to_have_text("0.6 s")
        playwright.expect(page.locator("#sandbox-start-duration")).to_have_text("0.0 s")
        playwright.expect(page.locator("#source-copy-duration")).to_have_text("Reused")
        playwright.expect(page.locator("#install-build-duration")).to_have_text("Reused")
        playwright.expect(page.locator("#install-build-details")).to_be_hidden()
        playwright.expect(page.locator("#install-build-metrics")).to_be_empty()
        playwright.expect(page.locator("#first-text-duration")).to_have_text("0.0 s")
        assert "1000.0 s" not in page.locator("#lifecycle-timings").inner_text()
    assert runtime.posts == []


def test_worker_summary_tracks_current_turn_and_clears_for_older_rows(ui):
    runtime, page, open_page = ui
    row = runtime.run("worker-session", run_id="cold-worker")
    runtime.complete(row)
    row["result"]["worker"] = {
        "mode": "persistent",
        "id": "0123456789abcdef",
        "pid": 101,
        "reused": False,
        "invocation_count": 1,
        "identity": "c" * 64,
    }
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'worker-session')")
    open_page()
    summary = page.locator("#worker-summary")
    playwright.expect(summary).to_have_text("Started worker · PID 101 · turn 1")
    playwright.expect(summary).to_have_attribute("title", "Worker 0123456789abcdef")
    warm = runtime.run("worker-session", run_id="warm-worker")
    runtime.complete(warm)
    warm["result"]["worker"] = {**row["result"]["worker"], "reused": True, "invocation_count": 2}
    page.locator("#refresh-session").click()
    playwright.expect(summary).to_have_text("Reused worker · PID 101 · turn 2")
    oneshot = runtime.run("worker-session", run_id="oneshot-worker")
    runtime.complete(oneshot)
    oneshot["result"]["worker"] = {"mode": "oneshot"}
    page.locator("#refresh-session").click()
    playwright.expect(summary).to_have_text("Fresh process")
    playwright.expect(summary).to_have_attribute("title", "")
    old = runtime.run("worker-session", run_id="no-worker-metadata")
    runtime.complete(old)
    page.locator("#refresh-session").click()
    page.wait_for_function("state.runs.length === 4 && !state.loading")
    playwright.expect(summary).to_be_hidden()
    playwright.expect(summary).to_be_empty()
    assert runtime.posts == []


def test_final_lifecycle_overrides_stale_running_snapshot(ui):
    runtime, page, open_page = ui
    row = runtime.run("stale-lifecycle")
    row["lifecycle"] = lifecycle(
        agent_run={"status": "running", "started_at": "2026-11-09T12:00:00Z"}
    )
    row["lifecycle"]["preparation_ms"] = 999999
    runtime.complete(row, "Successful reply")
    row["result"]["lifecycle"] = lifecycle(
        sandbox_start={"status": "completed", "duration_ms": 200},
        source_copy={"status": "completed", "duration_ms": 300},
        install_build={"status": "completed", "duration_ms": 700},
        agent_run={
            "status": "completed",
            "duration_ms": 8000,
            "details_ms": {"prepare": 5000, "agent_execution": 2000},
        },
    )
    row["result"]["lifecycle"]["preparation_ms"] = 1200
    page.add_init_script(
        "localStorage.setItem('sandbox-chat:session:agent-one', 'stale-lifecycle')"
    )
    open_page()
    playwright.expect(page.locator("#startup-duration")).to_have_text("1.2 s")
    playwright.expect(page.locator("#sandbox-start-duration")).to_have_text("0.2 s")
    playwright.expect(page.locator("#source-copy-duration")).to_have_text("0.3 s")
    playwright.expect(page.locator("#install-build-duration")).to_have_text("0.7 s")
    playwright.expect(page.locator("#agent-run-duration")).to_have_text("8.0 s")
    playwright.expect(page.locator("#step-agent-run")).to_have_attribute("data-status", "completed")
    page.locator("#agent-run-details summary").click()
    playwright.expect(page.locator("#agent-run-metrics")).to_contain_text("5.0 s")
    playwright.expect(page.locator("#chat-log .message.assistant")).to_contain_text(
        "Successful reply"
    )


@pytest.mark.parametrize("reused", [False, True])
def test_legacy_timings_keep_current_installation_separate_from_history(ui, reused):
    runtime, page, open_page = ui
    row = runtime.run("legacy-session")
    runtime.complete(row)
    row["result"].update(
        environment_reused=reused,
        timings_ms={
            "environment_startup": 3200,
            "registry_connect_and_lock": 100,
            "environment_probe": 200,
            "source_transfer": 700,
            "bootstrap": 2200,
            "worker_process_and_execution": 10000,
        },
        initial_bootstrap_timings_ms={"dependency_install": 1000},
    )
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'legacy-session')")
    open_page()
    playwright.expect(page.locator("#startup-duration")).to_have_text("3.2 s")
    playwright.expect(page.locator("#sandbox-start-duration")).to_have_text("0.3 s")
    playwright.expect(page.locator("#agent-run-duration")).to_have_text("10.0 s")
    playwright.expect(page.locator("#first-text-duration")).to_have_text("Unavailable")
    playwright.expect(page.locator("#source-copy-duration")).to_have_text(
        "Reused" if reused else "0.7 s"
    )
    playwright.expect(page.locator("#install-build-duration")).to_have_text(
        "Reused" if reused else "2.2 s"
    )
    if reused:
        playwright.expect(page.locator("#install-build-details")).to_be_hidden()
    else:
        page.locator("#install-build-details summary").click()
        playwright.expect(page.locator("#install-build-metrics")).to_contain_text("1.0 s")


def test_failed_installation_retains_completed_steps_and_partial_error(ui):
    runtime, page, open_page = ui
    row = runtime.run("build-failed", status="failed", stage="Failed", text="Partial output")
    row["error"] = "Installation failed"
    row["lifecycle"] = lifecycle(
        sandbox_start={"status": "completed", "duration_ms": 100},
        source_copy={"status": "completed", "duration_ms": 200},
        install_build={
            "status": "failed",
            "duration_ms": 4500,
            "details_ms": {"toolchain_download": 4000},
        },
    )
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'build-failed')")
    open_page()
    playwright.expect(page.locator("#sandbox-start-duration")).to_have_text("0.1 s")
    playwright.expect(page.locator("#source-copy-duration")).to_have_text("0.2 s")
    playwright.expect(page.locator("#install-build-duration")).to_have_text("Failed · 4.5 s")
    playwright.expect(page.locator("#agent-run-duration")).to_have_text("Not reached")
    playwright.expect(page.locator("#startup-duration")).to_have_text("Unavailable")
    playwright.expect(page.locator("#chat-log .message.error")).to_contain_text(
        "Installation failed"
    )
    playwright.expect(page.locator("#chat-log .message.assistant")).to_contain_text(
        "Partial output"
    )


@pytest.mark.parametrize(
    "status,label", [("interrupted", "Interrupted"), ("failed", "Unavailable")]
)
def test_terminal_running_steps_do_not_tick_after_restart(ui, status, label):
    runtime, page, open_page = ui
    row = runtime.run("interrupted-session", status=status, stage="Interrupted")
    row["lifecycle"] = lifecycle(
        sandbox_start={"status": "completed", "duration_ms": 100},
        source_copy={"status": "reused"},
        install_build={"status": "reused"},
        agent_run={"status": "running", "started_at": "2026-11-09T12:00:00Z"},
    )
    row["lifecycle"]["preparation_ms"] = 600
    page.add_init_script(
        "localStorage.setItem('sandbox-chat:session:agent-one', 'interrupted-session')"
    )
    open_page()
    playwright.expect(page.locator("#agent-run-duration")).to_have_text(label)
    playwright.expect(page.locator("#agent-run-detail")).to_contain_text(
        "before this timing was confirmed"
    )
    page.evaluate(
        "() => { const now = Date.now(); Date.now = () => now + 60000; renderLifecycle(); }"
    )
    playwright.expect(page.locator("#agent-run-duration")).to_have_text(label)
    assert "elapsed" not in page.locator("#lifecycle-timings").inner_text()


def test_new_chat_is_lazy_source_links_and_mobile(ui):
    runtime, page, open_page = ui
    open_page()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Not created")
    first = page.locator("#session-id").inner_text()
    page.locator("#new-session").click()
    playwright.expect(page.locator("#session-id")).not_to_have_text(first)
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Not created")
    assert runtime.posts == []
    assert (
        page.locator("#view-session-source").get_attribute("href") == "/api/agents/agent-one/source"
    )
    assert (
        page.locator("#open-source-folder").get_attribute("href")
        == "https://example.cloud.databricks.com/#workspace/Users/demo/A%20%23%3F%25%20%E6%97%A5%E6%9C%AC"
    )
    playwright.expect(page.locator("#active-git")).to_contain_text("feature/chat")
    page.set_viewport_size({"width": 375, "height": 812})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert page.locator("#composer").bounding_box()["width"] <= 375
    assert page.locator("#new-session").is_visible()


def test_failed_run_retry_creates_new_identity_and_retains_failed_history(ui):
    runtime, page, open_page = ui
    open_page()
    send(page)
    page.wait_for_function("state.runs.length === 1 && state.runs[0].status === 'queued'")
    row = runtime.runs[-1]
    for stage, label in [
        ("Creating sandbox", "Creating sandbox"),
        ("Uploading source", "Uploading source"),
        ("Installing dependencies and building agent", "Installing dependencies"),
        ("Running agent", "Running"),
    ]:
        row.update(status="running", stage=stage)
        playwright.expect(page.locator("#sandbox-status")).to_have_text(label, timeout=6000)
    assert len(runtime.sandbox_reads) == 1, (
        "Active progress must not repeatedly query cloud sandbox status"
    )
    row.update(
        status="failed",
        stage="Failed",
        stream={"text": "**Partial** response"},
        error="Model unavailable",
    )
    runtime.status[row["session_id"]] = "failed"
    playwright.expect(page.locator("#chat-log .message.assistant strong")).to_have_text(
        "Partial", timeout=6000
    )
    playwright.expect(page.locator("#chat-log .message.error")).to_contain_text("Model unavailable")
    playwright.expect(page.locator("#retry-run")).to_have_text("Retry this invocation")
    page.locator("#retry-run").click()
    page.wait_for_function("state.runs.length === 2 && state.runs[1].status === 'queued'")
    assert runtime.posts[0]["invocation_id"] != runtime.posts[1]["invocation_id"]
    assert runtime.posts[1]["retry"] is False
    assert runtime.posts[0]["input"] == runtime.posts[1]["input"]
    runtime.complete(runtime.runs[-1], "Final reply")
    playwright.expect(
        page.locator("#chat-log .message.assistant .message-content").last
    ).to_have_text("Final reply", timeout=6000)
    playwright.expect(page.locator("#chat-log .message.user")).to_have_count(2)
    playwright.expect(page.locator("#startup-duration")).to_have_text("2.3 s")
    playwright.expect(page.locator("#sandbox-id")).to_have_text(f"sandbox-{row['session_id']}")


def test_uncertain_post_reconnect_and_reload_do_not_create_new_invocation(ui):
    runtime, page, open_page = ui
    runtime.ambiguous = True
    open_page()
    send(page, "Potential side effect")
    playwright.expect(page.locator("#retry-run")).to_have_text("Reconnect this invocation")
    invocation = runtime.posts[0]["invocation_id"]
    page.locator("#retry-run").click()
    page.wait_for_function("state.runs.length === 1 && state.busy")
    assert [post["invocation_id"] for post in runtime.posts] == [invocation, invocation]
    assert not any(post["retry"] for post in runtime.posts)
    assert len(runtime.runs) == 1
    page.reload()
    playwright.expect(page.locator("#chat-log .message.user")).to_have_count(1)
    assert len(runtime.posts) == 2
    runtime.complete(runtime.runs[0])
    playwright.expect(page.locator("#chat-log .message.assistant strong")).to_have_text(
        "Finished", timeout=6000
    )


def test_pending_submission_survives_reload_before_reconnect(ui):
    runtime, page, open_page = ui
    runtime.ambiguous = True
    open_page()
    send(page)
    playwright.expect(page.locator("#retry-run")).to_be_visible()
    page.reload()
    # History discovers the already accepted invocation; no POST is necessary.
    playwright.expect(page.locator("#chat-log .message.user")).to_have_count(1)
    page.wait_for_function("state.busy && state.pending === null")
    assert len(runtime.posts) == 1


def test_actual_eventsource_snapshots_replace_text_and_safe_markdown(ui):
    runtime, page, open_page = ui
    runtime.event_stream = True
    row = runtime.run("persisted", text="Partial")
    partial = copy.deepcopy(row)
    runtime.complete(
        row,
        "**Final**\n\n<script>window.pwned=1</script>\n\n[unsafe](javascript:alert(1))\n\n![image](https://example.com/x.png)\n\n[safe](https://example.com)",
    )
    # History returns the active snapshot, then real EventSource dispatches partial + final.
    runtime.runs = [partial]
    runtime.events = [partial, row]
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'persisted')")
    open_page()
    playwright.expect(page.locator("#chat-log .message.assistant strong")).to_have_text("Final")
    playwright.expect(page.locator("#chat-log .message.assistant")).to_have_count(1)
    assert page.evaluate("window.pwned") is None
    assert (
        page.locator("#chat-log script, #chat-log img, #chat-log a[href^='javascript:']").count()
        == 0
    )
    assert all(
        link.get_attribute("rel") == "noopener noreferrer"
        for link in page.locator("#chat-log a").all()
    )
    assert runtime.posts == []


def test_custom_callable_final_json_replaces_partial_stream_and_reloads(ui):
    runtime, page, open_page = ui
    runtime.event_stream = True
    row = runtime.run("custom-result", text="Stale partial stream")
    partial = copy.deepcopy(row)
    runtime.complete(row)
    row["result"]["result"] = {"count": 1}
    runtime.runs = [partial]
    runtime.events = [partial, row]
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'custom-result')")
    open_page()
    answer = page.locator("#chat-log .message.assistant .message-content")
    playwright.expect(answer).to_have_text('{ "count": 1 }')
    assert answer.get_attribute("data-source") == json.dumps({"count": 1}, indent=2)
    assert "Stale partial stream" not in page.locator("#chat-log").inner_text()
    playwright.expect(page.locator("#chat-log .message.assistant")).to_have_count(1)
    runtime.runs = [row]
    page.reload()
    playwright.expect(answer).to_have_text('{ "count": 1 }')
    assert runtime.posts == []


def test_legacy_reasoning_display_override_applies_to_stream_history_and_copy(ui):
    runtime, page, open_page = ui
    raw = '[{"type": "reasoning", "summary": [{"signature": "test-signature", "text": ""}]}]'
    row = runtime.run("legacy", text=raw)
    row["display_text"] = ""
    page.add_init_script("""window.sources = []; window.EventSource = class extends EventTarget {
      constructor() { super(); window.sources.push(this); } close() {} };
      localStorage.setItem('sandbox-chat:session:agent-one', 'legacy');
      Object.defineProperty(navigator, 'clipboard', {value: {writeText: async text => {window.copied = text;}}});
    """)
    open_page()
    assert "test-signature" not in page.locator("#chat-log").inner_text()
    row["stream"]["text"] = raw + "Partial"
    row["display_text"] = "Partial"
    page.evaluate(
        "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))",
        row,
    )
    answer = page.locator("#chat-log .message.assistant .message-content")
    playwright.expect(answer).to_have_text("Partial")
    runtime.complete(row, raw + "**Final answer**")
    row["display_text"] = "**Final answer**"
    page.evaluate(
        "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))",
        row,
    )
    playwright.expect(answer).to_have_text("Final answer")
    page.locator("#chat-log .message.assistant button").click()
    assert page.evaluate("window.copied") == "**Final answer**"
    page.reload()
    playwright.expect(answer).to_have_text("Final answer")
    assert "test-signature" not in page.locator("#chat-log").inner_text()
    assert runtime.posts == []


def test_terminal_sandbox_waits_for_delayed_status_without_not_created_flash(ui):
    runtime, page, open_page = ui
    page.add_init_script("""window.sources = []; window.EventSource = class extends EventTarget {
      constructor() { super(); window.sources.push(this); } close() {} };""")
    open_page()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Not created")
    send(page)
    page.wait_for_function("window.sources.length === 1")
    row = runtime.runs[-1]
    row.update(status="running", stage="Running agent")
    dispatch = "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))"
    page.evaluate(dispatch, row)
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Running")
    held = []
    page.route(
        lambda url: urlsplit(url).path.endswith("/sandbox"), lambda route: held.append(route)
    )
    runtime.complete(row)
    page.evaluate(dispatch, row)
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Checking sandbox…")
    playwright.expect(page.locator("#sandbox-id")).to_have_text(f"sandbox-{row['session_id']}")
    playwright.expect(page.locator("#stop-session")).to_be_disabled()
    assert len(held) == 1
    runtime.route(held.pop())
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Running")
    playwright.expect(page.locator("#stop-session")).to_be_enabled()
    assert len(runtime.posts) == 1


def test_session_switch_ignores_late_events_and_stop_is_scoped(ui):
    runtime, page, open_page = ui
    row = runtime.run("old-session", text="Old answer")
    runtime.status["new-session"] = "running"
    runtime.run("new-session", run_id="run-two", status="succeeded")
    page.add_init_script("""window.sources = []; window.EventSource = class extends EventTarget {
      constructor() { super(); window.sources.push(this); } close() {} };
      localStorage.setItem('sandbox-chat:session:agent-one', 'old-session');""")
    open_page()
    page.wait_for_function("window.sources.length === 1")
    page.locator(".session-open-button", has_text="new-session").click()
    playwright.expect(page.locator("#session-id")).to_have_text("new-session")
    playwright.expect(page.locator("#sandbox-id")).to_have_text("sandbox-new-session")
    late = {**row, "stream": {"text": "Stale response must not appear"}}
    page.evaluate(
        "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))",
        late,
    )
    assert "Stale response" not in page.locator("#chat-log").inner_text()
    page.locator("#stop-session").click()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Stopped")
    playwright.expect(page.locator("#stop-duration")).to_have_text("1.5 s")
    assert runtime.stops == [{"agent_id": "agent-one", "session_id": "new-session"}]
    page.locator(".session-open-button", has_text="old-session").click()
    playwright.expect(page.locator("#session-id")).to_have_text("old-session")
    playwright.expect(page.locator("#stop-duration")).to_have_text("—")


def test_failed_history_blocks_send_and_reconnect_restores_it(ui):
    runtime, page, open_page = ui
    open_page()
    runtime.history_error = True
    page.locator("#refresh-session").click()
    playwright.expect(page.locator("#request-notice")).to_contain_text(
        "Could not load conversation"
    )
    page.locator("#prompt-input").fill("Do not submit without history")
    playwright.expect(page.locator("#send-button")).to_be_disabled()
    playwright.expect(page.locator("#retry-run")).to_be_enabled()
    runtime.history_error = False
    page.locator("#retry-run").click()
    playwright.expect(page.locator("#request-notice")).to_be_hidden()
    assert runtime.posts == []


def test_generic_revision_selection_and_legacy_scope(ui):
    runtime, page, open_page = ui
    runtime.config.update(
        default_environment=None, default_agent_id=None, sandbox_scope="environment"
    )
    runtime.agent["source_path"] = "/Volumes/catalog/schema/data"
    runtime.agent["source_kind"] = "volume"
    open_page()
    playwright.expect(page.locator("#source-selection")).to_be_visible()
    playwright.expect(page.locator("#open-source-folder")).to_be_hidden()
    playwright.expect(page.locator("#scope-description")).to_contain_text("share one sandbox")
    playwright.expect(page.locator("#stop-session")).to_have_text("Stop shared sandbox")
    page.locator("#source-selection").evaluate("element => element.open = true")
    page.locator("#agent-name").fill("New source")
    page.locator("#source-kind").select_option("bundled")
    page.locator("#load-source").click()
    playwright.expect(page.locator("#agent-title")).to_have_text("New source")
    assert runtime.posts == []
    assert (
        page.locator("#view-session-source").get_attribute("href") == "/api/agents/new-agent/source"
    )
    send(page)
    page.wait_for_function("state.runs.length === 1")
    assert runtime.posts[0]["agent_id"] == "new-agent"


def test_normalized_workspace_links_and_unsafe_host(ui):
    runtime, page, open_page = ui
    runtime.agent["source_path"] = "/Users/demo/A #?% 日本"
    open_page()
    assert page.locator("#open-source-folder").get_attribute("href") == (
        "https://example.cloud.databricks.com/#workspace/Users/demo/A%20%23%3F%25%20%E6%97%A5%E6%9C%AC"
    )
    assert page.evaluate("workingFolderUrl('javascript:alert(1)', '/Users/demo')") is None
    assert (
        page.evaluate("workingFolderUrl('https://user:password@example.com', '/Users/demo')")
        is None
    )
    assert page.evaluate("workingFolderUrl('https://example.com', '//external.example')") is None


def test_uncertain_failed_retry_reconnect_cannot_requeue_again(ui):
    runtime, page, open_page = ui
    row = runtime.run("failed-session", status="failed", stage="Failed", text="Partial")
    row["error"] = "Invocation failed"
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'failed-session')")
    open_page()
    runtime.ambiguous = True
    page.locator("#retry-run").click()
    playwright.expect(page.locator("#retry-run")).to_have_text("Reconnect this invocation")
    # The explicit retry reached the server, failed again, and its response was lost.
    runtime.runs[-1].update(status="failed", stage="Failed", error="Failed again")
    page.locator("#retry-run").click()
    playwright.expect(page.locator("#retry-run")).to_have_text("Retry this invocation")
    assert len(runtime.runs) == 2
    assert [post["retry"] for post in runtime.posts] == [False, False]
    assert len({post["invocation_id"] for post in runtime.posts}) == 1


def test_stop_timeout_is_unconfirmed_and_reconnect_only_reads(ui):
    runtime, page, open_page = ui
    row = runtime.run("stop-session", status="succeeded", stage="Complete")
    runtime.status[row["session_id"]] = "running"
    runtime.stop_error = True
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'stop-session')")
    open_page()
    page.locator("#stop-session").click()
    playwright.expect(page.locator("#request-notice")).to_contain_text("Stop was not confirmed")
    playwright.expect(page.locator("#stop-duration")).to_contain_text("unconfirmed")
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Running")
    page.locator("#retry-run").click()
    playwright.expect(page.locator("#request-notice")).to_be_hidden()
    assert len(runtime.stops) == 1
    assert runtime.posts == []


def test_delayed_stop_blocks_duplicates_and_send_and_needs_no_followup_read(ui):
    runtime, page, open_page = ui
    row = runtime.run("slow-stop", status="succeeded", stage="Complete")
    runtime.status[row["session_id"]] = "running"
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'slow-stop')")
    open_page()
    held = []
    page.route("**/sessions/stop", lambda route: held.append(route))
    page.locator("#stop-session").click()
    playwright.expect(page.locator("#stop-session")).to_have_text("Stopping…")
    playwright.expect(page.locator("#stop-session")).to_be_disabled()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Stopping…")
    playwright.expect(page.locator("#stop-duration")).to_contain_text("elapsed")
    page.locator("#events-tab").click()
    playwright.expect(page.locator("#sandbox-tab-badge")).to_have_text("Stopping")
    page.locator("#prompt-input").fill("Wait for the stop")
    playwright.expect(page.locator("#send-button")).to_be_disabled()
    page.evaluate("void stopSession(); document.getElementById('composer').requestSubmit()")
    assert len(held) == 1
    assert runtime.posts == []
    status_reads = len(runtime.sandbox_reads)
    runtime.route(held.pop())
    page.locator("#sandbox-tab").click()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Stopped")
    playwright.expect(page.locator("#stop-duration")).to_have_text("1.5 s")
    playwright.expect(page.locator("#stop-session")).to_have_text("Stop this session")
    playwright.expect(page.locator("#send-button")).to_be_enabled()
    assert len(runtime.sandbox_reads) == status_reads
    assert runtime.stops == [{"agent_id": "agent-one", "session_id": "slow-stop"}]


def test_confirmed_stop_fences_delayed_running_status_read(ui):
    runtime, page, open_page = ui
    row = runtime.run("stale-stop", status="succeeded", stage="Complete")
    runtime.status[row["session_id"]] = "running"
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'stale-stop')")
    open_page()
    held = []
    page.route(
        lambda url: urlsplit(url).path.endswith("/sandbox"), lambda route: held.append(route)
    )
    with page.expect_request("**/sandbox?*"):
        page.evaluate("void (window.pendingStatus = refreshSandbox(state.context))")
    page.locator("#stop-session").click()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Stopped")
    assert len(held) == 1
    held.pop().fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps({"sandbox_id": "sandbox-stale-stop", "status": "running"}),
    )
    page.evaluate("window.pendingStatus")
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Stopped")
    playwright.expect(page.locator("#stop-session")).to_be_disabled()
    assert len(runtime.stops) == 1


@pytest.mark.parametrize("return_before_confirmation", [True, False])
def test_pending_stop_survives_switching_sessions(ui, return_before_confirmation):
    runtime, page, open_page = ui
    for index, session in enumerate(("stop-a", "stop-b")):
        runtime.run(session, run_id=f"run-{index}", status="succeeded", stage="Complete")
        runtime.status[session] = "running"
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'stop-a')")
    open_page()
    held = []
    page.route("**/sessions/stop", lambda route: held.append(route))
    page.locator("#stop-session").click()
    page.locator(".session-open-button", has_text="stop-b").click()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Running")
    playwright.expect(page.locator("#stop-session")).to_be_enabled()
    if return_before_confirmation:
        page.locator(".session-open-button", has_text="stop-a").click()
        playwright.expect(page.locator("#stop-session")).to_have_text("Stopping…")
        playwright.expect(page.locator("#stop-session")).to_be_disabled()
        page.locator("#prompt-input").fill("Do not resume while stopping")
        playwright.expect(page.locator("#send-button")).to_be_disabled()
        page.evaluate("void stopSession(); document.getElementById('composer').requestSubmit()")
    assert len(held) == 1
    runtime.route(held.pop())
    page.wait_for_function("state.stopOperations.size === 0")
    if not return_before_confirmation:
        playwright.expect(page.locator("#sandbox-status")).to_have_text("Running")
        playwright.expect(page.locator("#stop-duration")).to_have_text("—")
        page.locator(".session-open-button", has_text="stop-a").click()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Stopped")
    playwright.expect(page.locator("#stop-duration")).to_have_text("1.5 s")
    page.locator(".session-open-button", has_text="stop-b").click()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Running")
    playwright.expect(page.locator("#stop-session")).to_be_enabled()
    assert runtime.stops == [{"agent_id": "agent-one", "session_id": "stop-a"}]
    assert runtime.posts == []


def test_stop_disabled_reasons_cover_no_sandbox_pending_and_active_invocation(ui):
    runtime, page, open_page = ui
    page.add_init_script("""window.sources = []; window.EventSource = class extends EventTarget {
      constructor() { super(); window.sources.push(this); } close() {} };""")
    open_page()
    button = page.locator("#stop-session")
    playwright.expect(button).to_be_disabled()
    playwright.expect(page.locator("#stop-reason")).to_contain_text("Send a message")
    assert button.evaluate("element => getComputedStyle(element).cursor") == "default"
    assert float(button.evaluate("element => getComputedStyle(element).opacity")) < 1
    runtime.ambiguous = True
    send(page)
    playwright.expect(page.locator("#stop-reason")).to_contain_text(
        "Confirm the pending invocation"
    )
    page.locator("#retry-run").click()
    playwright.expect(page.locator("#stop-reason")).to_contain_text("invocation is in progress")
    playwright.expect(button).to_be_disabled()
    row = runtime.runs[-1]
    runtime.complete(row)
    row["result"]["worker"] = {"mode": "persistent", "reused": True, "pid": 123}
    page.evaluate(
        "row => window.sources[0].dispatchEvent(new MessageEvent('run', {data: JSON.stringify(row)}))",
        row,
    )
    playwright.expect(button).to_be_enabled()
    playwright.expect(page.locator("#stop-reason")).to_be_hidden()
    assert runtime.stops == []


def test_stop_fetch_deadline_is_unconfirmed_and_reconnect_does_not_replay(ui):
    runtime, page, open_page = ui
    runtime.run("deadline-stop", status="succeeded", stage="Complete")
    runtime.status["deadline-stop"] = "running"
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'deadline-stop')")
    open_page()
    held = []
    page.route("**/sessions/stop", lambda route: held.append(route))
    page.evaluate("API_TIMEOUTS.stop = 100")
    page.locator("#stop-session").click()
    playwright.expect(page.locator("#request-notice")).to_contain_text("Request timed out")
    playwright.expect(page.locator("#stop-duration")).to_contain_text("unconfirmed")
    page.wait_for_function("state.stopOperations.size === 0")
    page.locator("#retry-run").click()
    playwright.expect(page.locator("#request-notice")).to_be_hidden()
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Running")
    assert len(held) == 1
    assert runtime.posts == []


def test_status_read_deadline_preserves_known_id_and_unblocks_session_loading(ui):
    runtime, page, open_page = ui
    row = runtime.run("deadline-status", status="succeeded", stage="Complete")
    runtime.complete(row)
    page.add_init_script(
        "localStorage.setItem('sandbox-chat:session:agent-one', 'deadline-status')"
    )
    open_page()
    held = []
    page.route(
        lambda url: urlsplit(url).path.endswith("/sandbox"), lambda route: held.append(route)
    )
    page.evaluate("API_TIMEOUTS.status = 100")
    page.locator("#refresh-session").click()
    playwright.expect(page.locator("#request-notice")).to_contain_text("Request timed out")
    page.wait_for_function("!state.loading")
    playwright.expect(page.locator("#sandbox-status")).to_have_text("Unknown")
    playwright.expect(page.locator("#sandbox-id")).to_have_text("sandbox-deadline-status")
    playwright.expect(page.locator("#stop-session")).to_be_enabled()
    assert len(held) == 1
    assert runtime.stops == runtime.posts == []


def test_pending_identity_storage_failure_prevents_submission(ui):
    runtime, page, open_page = ui
    open_page()
    page.evaluate(
        "() => { Storage.prototype.setItem = () => { throw new Error('Storage blocked'); }; }"
    )
    page.locator("#prompt-input").fill("Do not lose this invocation identity")
    page.locator("#send-button").click()
    playwright.expect(page.locator("#request-notice")).to_contain_text(
        "Browser storage is unavailable"
    )
    assert runtime.posts == []


def test_new_chat_prepares_before_send_and_shows_handler_initialization(ui):
    runtime, page, open_page = ui
    runtime.config["capabilities"] = ["session-preparation"]
    open_page()
    page.locator("#new-session").click()
    playwright.expect(page.locator("#sandbox-tab")).to_have_attribute("aria-selected", "true")
    playwright.expect(page.locator("#run-status")).to_have_text("Waiting to prepare session")
    playwright.expect(page.locator("#session-readiness")).to_be_visible()
    assert (
        page.locator(".readiness-spinner").evaluate("e => getComputedStyle(e).animationName")
        == "readiness-spin"
    )
    page.locator("#prompt-input").fill("hello")
    playwright.expect(page.locator("#send-button")).to_be_disabled()
    assert runtime.posts == []
    assert len(runtime.preparation_posts) == 1
    session = runtime.preparation_posts[0]["session_id"]
    runtime.preparations[session].update(
        status="preparing",
        stage="Initializing resident handler",
        lifecycle={
            "steps": {
                "handler_prepare": {
                    "status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        },
    )
    playwright.expect(page.locator("#run-status")).to_have_text(
        "Initializing resident handler", timeout=6000
    )
    playwright.expect(page.locator("#handler-prepare-duration")).to_contain_text("Running")
    runtime.status[session] = "running"
    page.evaluate("refreshSandbox(state.context)")
    playwright.expect(page.locator("#send-button")).to_be_disabled()
    playwright.expect(page.locator("#session-readiness")).to_be_visible()
    runtime.preparations[session].update(
        status="ready",
        stage="Session prepared",
        lifecycle={
            "steps": {
                "handler_prepare": {"status": "completed", "duration_ms": 1234},
                "agent_run": {"status": "skipped"},
            },
            "preparation_ms": 10000,
        },
    )
    playwright.expect(page.locator("#send-button")).to_be_enabled(timeout=6000)
    playwright.expect(page.locator("#session-readiness")).to_be_hidden()
    playwright.expect(page.locator("#handler-prepare-duration")).to_have_text("1.2 s")
    playwright.expect(page.locator("#first-text-duration")).to_have_text("—")
    playwright.expect(page.locator("#startup-detail")).to_contain_text("handler initialization")
    page.locator("#send-button").click()
    playwright.expect(page.locator("#chat-log .message.user")).to_have_count(1)
    assert runtime.posts[0]["session_id"] == session


def test_failed_session_preparation_can_be_retried_without_an_invocation(ui):
    runtime, page, open_page = ui
    runtime.config["capabilities"] = ["session-preparation"]
    open_page()
    page.locator("#new-session").click()
    playwright.expect(page.locator("#run-status")).to_have_text("Waiting to prepare session")
    session = runtime.preparation_posts[0]["session_id"]
    runtime.preparations[session].update(status="failed", error="Dependency installation failed")
    playwright.expect(page.locator("#request-notice")).to_have_text(
        "Dependency installation failed", timeout=6000
    )
    playwright.expect(page.locator("#prepare-session")).to_have_text("Retry preparation")
    page.locator("#prepare-session").click()
    playwright.expect(page.locator("#run-status")).to_have_text("Waiting to prepare session")
    assert len(runtime.preparation_posts) == 2
    assert {row["session_id"] for row in runtime.preparation_posts} == {session}
    assert runtime.posts == []


def test_resume_spinner_waits_for_worker_and_survives_reload(ui):
    runtime, page, open_page = ui
    runtime.config["capabilities"] = ["session-preparation"]
    row = runtime.run("resume-me", status="succeeded")
    runtime.complete(row)
    runtime.status["resume-me"] = "stopped"
    runtime.preparations["resume-me"] = {"status": "ready", "stage": "Previous preparation"}
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'resume-me')")
    open_page()
    playwright.expect(page.locator("#prepare-session")).to_have_text("Resume session")
    page.locator("#prepare-session").click()
    playwright.expect(page.locator("#session-readiness")).to_be_visible()
    runtime.status["resume-me"] = "running"
    runtime.preparations["resume-me"].update(
        status="preparing", stage="Initializing resident handler"
    )
    page.reload()
    playwright.expect(page.locator("#readiness-message")).to_have_text(
        "Initializing resident handler"
    )
    page.locator("#prompt-input").fill("Resume this conversation")
    playwright.expect(page.locator("#send-button")).to_be_disabled()
    runtime.preparations["resume-me"].update(status="ready", stage="Session prepared")
    playwright.expect(page.locator("#session-readiness")).to_be_hidden(timeout=6000)
    playwright.expect(page.locator("#send-button")).to_be_enabled()
    assert len(runtime.preparation_posts) == 1
    assert not runtime.posts


def test_running_sandbox_without_worker_readiness_cannot_enable_send(ui):
    runtime, page, open_page = ui
    runtime.config["capabilities"] = ["session-preparation"]
    runtime.status["needs-worker"] = "running"
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'needs-worker')")
    open_page()
    page.locator("#prompt-input").fill("Wait for the worker")
    playwright.expect(page.locator("#send-button")).to_be_disabled()
    playwright.expect(page.locator("#prepare-session")).to_be_enabled()
    assert not runtime.preparation_posts
    assert not runtime.posts


def test_stream_batches_render_progressively_and_terminal_flushes_without_delay(ui):
    _, page, open_page = ui
    open_page()
    result = page.evaluate("""async () => {
        const row = {id: 'buffer-test', input: {message: 'Hello'}, status: 'running',
            stream: {text: ''}, created_at: new Date().toISOString()};
        mergeRun(row);
        const element = document.querySelector('.message.assistant .message-content');
        const updates = [];
        const observer = new MutationObserver(() => updates.push(element.dataset.source));
        observer.observe(element, {childList: true, subtree: true, characterData: true});
        const text = 'Hello 🌍. '.repeat(30);
        mergeRun({...row, stream: {text}});
        await new Promise(resolve => setTimeout(resolve, 160));
        const completedBatch = element.dataset.source;
        mergeRun({...row, stream: {text: text + 'queued display'}});
        mergeRun({...row, status: 'succeeded', result: {result: {reply: text + 'Final'}}});
        const terminalText = element.dataset.source;
        await new Promise(resolve => setTimeout(resolve, 100));
        observer.disconnect();
        return {updates, completedBatch, text, terminalText, after: element.dataset.source};
    }""")
    assert result["completedBatch"] == result["text"]
    assert any(0 < len(text) < len(result["text"]) for text in result["updates"])
    assert all(not any(0xD800 <= ord(c) <= 0xDFFF for c in text) for text in result["updates"])
    assert result["terminalText"] == result["after"] == result["text"] + "Final"


def test_reduced_motion_streaming_renders_received_text_immediately(ui):
    _, page, open_page = ui
    page.emulate_media(reduced_motion="reduce")
    open_page()
    assert (
        page.evaluate("""() => {
        mergeRun({id: 'accessible-stream', input: {}, status: 'running', stream: {text: 'Received text'}});
        return document.querySelector('.message.assistant .message-content').dataset.source;
    }""")
        == "Received text"
    )


def test_stream_transport_reconnects_without_submitting_another_invocation(ui):
    runtime, page, open_page = ui
    active = runtime.run("reconnect", text="Hello")
    runtime.events = [copy.deepcopy(active)]
    runtime.event_stream = True
    page.add_init_script("localStorage.setItem('sandbox-chat:session:agent-one', 'reconnect')")
    open_page()
    playwright.expect(page.locator("#chat-log .message.assistant")).to_contain_text("Hello")
    # The finite first SSE response closes. Next connection replays a full
    # snapshot and completes; history/status polling alone cannot see this answer.
    final = copy.deepcopy(active)
    runtime.complete(final, "Hello after reconnect")
    runtime.events = [final]
    playwright.expect(page.locator("#chat-log .message.assistant")).to_contain_text(
        "Hello after reconnect", timeout=6000
    )
    assert runtime.posts == []
    playwright.expect(page.locator("#chat-log .message.assistant")).to_have_count(1)
