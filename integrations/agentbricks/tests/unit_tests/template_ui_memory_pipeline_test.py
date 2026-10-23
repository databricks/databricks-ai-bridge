"""The chat app's memory pipeline block: demo routes and the shipped Memory pane JavaScript."""

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from databricks_agentkit import DurableAgentServer
from databricks_agentkit.runtime.auth import InvocationAuthPolicy
from databricks_agentkit.runtime.store import InMemoryRuntimeStore

UI_TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"
FRAMEWORKS = ["openai", "langgraph"]
PIPELINE = {
    "name": "memory-pipelines/p-1",
    "display_name": "dreaming-agent-memory",
    "memory_store": "memory-stores/store",
    "session_store": "session-stores/sessions",
    "dreamer_policy": {
        "model": "system.ai.claude-sonnet-4-5",
        "trigger": "MANUAL_ONLY",
        "instructions": "Keep only durable preferences.",
    },
}


def _load_ui(framework):
    path = UI_TEMPLATES / f"agent-{framework}" / "runtime" / "ui.py"
    spec = importlib.util.spec_from_file_location(f"_pipeline_ui_{framework}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ApiError(RuntimeError):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


class _PipelineClient:
    def __init__(self, pipeline=PIPELINE, runs=(), run_error=None):
        self.pipeline = pipeline
        self.runs = list(runs)
        self.run_error = run_error
        self.calls = []

    def find_memory_pipeline(self):
        self.calls.append(("find",))
        return self.pipeline

    def list_memory_pipeline_runs(self, pipeline):
        self.calls.append(("runs", pipeline))
        return {"pipeline_runs": self.runs}

    def run_memory_pipeline(self, pipeline):
        self.calls.append(("run", pipeline))
        if self.run_error:
            raise self.run_error
        return {"name": f"{pipeline}/runs/9", "state": "RUNNING"}


@pytest.fixture(params=FRAMEWORKS)
def ui(request, monkeypatch):
    module = _load_ui(request.param)
    monkeypatch.setattr(module, "_default_model", lambda: "system.ai.claude-sonnet-4-5")
    for name in ("AGENT_MEMORY_STORE", "AGENT_SESSION_STORE", "DATABRICKS_APP_NAME"):
        monkeypatch.delenv(name, raising=False)
    return module


def _client(ui, monkeypatch, state_client, *, memory=True, sessions=True, deployed=True):
    if memory:
        monkeypatch.setenv("AGENT_MEMORY_STORE", "store")
    if sessions:
        monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    if deployed:
        monkeypatch.setenv("DATABRICKS_APP_NAME", "agent")
        monkeypatch.setenv("DATABRICKS_APP_URL", "https://agent.example.databricksapps.com")
    monkeypatch.setattr(ui, "_state_client", lambda: state_client)

    async def invoke_handler(request, context):
        return {"output": []}

    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(), auth_policy=InvocationAuthPolicy()
    )
    app.invoke(invoke_handler)
    app.recover(invoke_handler)
    ui.install_ui(app)
    client = TestClient(app, base_url="https://testserver")
    client.params = {"session_id": "s1"}
    return client


@pytest.mark.parametrize(
    ("memory", "sessions", "deployed"),
    [(False, False, True), (True, False, True), (False, True, True)],
)
def test_pipeline_routes_need_both_stores_and_make_no_calls(
    ui, monkeypatch, memory, sessions, deployed
):
    state_client = _PipelineClient()
    client = _client(
        ui, monkeypatch, state_client, memory=memory, sessions=sessions, deployed=deployed
    )

    assert client.get("/api/ui/config").json()["memory"]["pipelines"] is False
    for response in (
        client.get("/api/demo/memory/pipeline"),
        client.post("/api/demo/memory/pipeline/run"),
    ):
        assert response.status_code == 503
        assert response.json()["detail"] == (
            "Memory pipelines need a deployed app with sessions and memory bound."
        )
    assert state_client.calls == []


def test_pipelines_are_off_in_local_dev_even_with_stores_bound(ui, monkeypatch):
    client = _client(ui, monkeypatch, _PipelineClient(), deployed=False)

    assert client.get("/api/ui/config").json()["memory"]["pipelines"] is False


def test_pipeline_route_returns_the_bound_pipeline_and_five_newest_runs(ui, monkeypatch):
    runs = [
        {"name": f"memory-pipelines/p-1/runs/{day}", "create_time": f"2026-10-{day:02d}T09:00:00Z"}
        for day in (14, 20, 16, 18, 15, 19, 17)
    ]
    runs[1].update(state="FAILED", error="Model service not found")
    state_client = _PipelineClient(runs=runs)
    client = _client(ui, monkeypatch, state_client)

    assert client.get("/api/ui/config").json()["memory"]["pipelines"] is True
    body = client.get("/api/demo/memory/pipeline").json()

    assert body["pipeline"] == PIPELINE
    assert [run["name"].rsplit("/", 1)[1] for run in body["runs"]] == ["20", "19", "18", "17", "16"]
    assert body["runs"][0]["state"] == "FAILED"
    assert body["runs"][0]["error"] == "Model service not found"
    assert state_client.calls == [("find",), ("runs", "memory-pipelines/p-1")]


def test_pipeline_route_reports_no_pipeline_for_the_pair(ui, monkeypatch):
    client = _client(ui, monkeypatch, _PipelineClient(pipeline=None))

    assert client.get("/api/demo/memory/pipeline").json() == {"pipeline": None, "runs": []}
    run = client.post("/api/demo/memory/pipeline/run")
    assert run.status_code == 404
    assert run.json()["detail"] == "No pipeline for this store pair."


def test_run_route_starts_the_bound_pipeline(ui, monkeypatch):
    state_client = _PipelineClient()
    client = _client(ui, monkeypatch, state_client)

    response = client.post("/api/demo/memory/pipeline/run")

    assert response.status_code == 200
    assert response.json() == {"name": "memory-pipelines/p-1/runs/9", "state": "RUNNING"}
    assert state_client.calls == [("find",), ("run", "memory-pipelines/p-1")]


def test_run_route_maps_a_run_in_progress_to_409(ui, monkeypatch):
    active = "memory-pipelines/p-1/runs/8"
    error = _ApiError(f"memory-pipelines/p-1 already has a run in progress: {active}.", "ABORTED")
    client = _client(ui, monkeypatch, _PipelineClient(run_error=error))

    response = client.post("/api/demo/memory/pipeline/run")

    assert response.status_code == 409
    assert response.json() == {"detail": "A run is already in progress.", "active_run": active}


def test_run_route_surfaces_other_api_errors(ui, monkeypatch):
    error = _ApiError("caller cannot run this pipeline", "PERMISSION_DENIED")
    client = _client(ui, monkeypatch, _PipelineClient(run_error=error))

    response = client.post("/api/demo/memory/pipeline/run")

    assert response.status_code == 502
    assert response.json()["detail"] == "PERMISSION_DENIED: caller cannot run this pipeline"


def test_find_memory_pipeline_matches_the_bound_pair_across_pages(ui, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_STORE", "store")
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    other_pair = {**PIPELINE, "name": "memory-pipelines/p-0", "session_store": "session-stores/x"}
    pages = [
        {"memory_pipelines": [other_pair], "next_page_token": "t2"},
        {"memory_pipelines": [PIPELINE], "next_page_token": "t3"},
    ]
    calls = []
    state_client = object.__new__(ui._ManagedStateClient)
    state_client._do = (
        lambda method, path, **kwargs: calls.append((method, path, kwargs)) or pages[len(calls) - 1]
    )

    assert state_client.find_memory_pipeline() == PIPELINE
    assert calls == [
        ("GET", "/api/2.0/agents/memory-pipelines", {"query": {"page_size": 100}}),
        (
            "GET",
            "/api/2.0/agents/memory-pipelines",
            {"query": {"page_size": 100, "page_token": "t2"}},
        ),
    ]


def test_find_memory_pipeline_stops_at_the_page_cap(ui, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_STORE", "store")
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    calls = []
    state_client = object.__new__(ui._ManagedStateClient)
    state_client._do = lambda method, path, **kwargs: calls.append(path) or {
        "memory_pipelines": [],
        "next_page_token": "more",
    }

    assert state_client.find_memory_pipeline() is None
    assert len(calls) == ui._PIPELINE_LIST_PAGES


def test_pipeline_run_calls_use_the_pipeline_resource(ui):
    calls = []
    state_client = object.__new__(ui._ManagedStateClient)
    state_client._do = lambda method, path, **kwargs: calls.append((method, path, kwargs)) or {}

    state_client.list_memory_pipeline_runs("memory-pipelines/p-1")
    state_client.run_memory_pipeline("memory-pipelines/p-1")

    assert calls == [
        ("GET", "/api/2.0/agents/memory-pipelines/p-1/runs", {"query": {"page_size": 5}}),
        ("POST", "/api/2.0/agents/memory-pipelines/p-1/run", {"body": {}}),
    ]
    with pytest.raises(ValueError):
        state_client.run_memory_pipeline("memory-pipelines/p-1/../../memory-stores/store")


# ---- Memory pane JavaScript ------------------------------------------------

_JS_FUNCTIONS = (
    "formatJson prettyDateTime memoryStoreName jsonResponse pipelineText pipelineCommand "
    "pipelineSetting pipelineRunError pipelineRunning updatePipelineButton setPipelineConfirm "
    "renderPipeline loadPipeline runPipeline"
).split()

_JS_HARNESS = r"""
import assert from 'node:assert/strict';
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.textContent = ''; this.hidden = false;
    this.disabled = false; this.className = ''; this.attrs = {}; this.listeners = {};
    this.classList = { toggle: () => {} }; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  set innerHTML(value) {}
  text() { return [this.textContent, ...this.children.map((child) => child.text())].join(' '); }
}
const document = { createElement: (tag) => new Node(tag) };
const elements = { memoryPipelineRun: new Node('button'), memoryPipelineBody: new Node('div'),
  memoryPipelineConfirm: new Node('div'), memoryPipelineConfirmText: new Node('p'),
  memoryPane: new Node('div') };
elements.memoryPipelineConfirm.hidden = true;
const state = { config: { memory: { pipelines: true, store: 'memory-stores/store' },
  session: { store: 'sessions' } } };
const events = [], requests = [], timers = [], log = [];
const addEvent = (type, payload) => events.push({ type, payload });
const demoUrl = (path) => path;
let responses = {};
const window = {
  setTimeout: (fn, ms) => { timers.push(ms); return timers.length; },
  clearTimeout: () => {},
};
const fetch = async (url, options = {}) => {
  const method = options.method || 'GET';
  requests.push(`${method} ${url}`);
  log.push(['fetch', `${method} ${url}`]);
  const [status, body] = responses[`${method} ${url}`];
  return { ok: status < 400, status, json: async () => body };
};
const navigator = { clipboard: { writeText: async () => {} } };
const GET = 'GET /api/demo/memory/pipeline', POST = 'POST /api/demo/memory/pipeline/run';
const pipeline = JSON.parse(process.argv[1]);
const body = () => elements.memoryPipelineBody.text();
const button = elements.memoryPipelineRun;
const confirmBox = elements.memoryPipelineConfirm;

// FR-6: without pipelines in config the pane says so and makes no calls.
state.config.memory.pipelines = false;
await loadPipeline();
assert.equal(requests.length, 0);
assert.match(body(), /Memory pipelines need a deployed app with sessions and memory bound\./);
assert.equal(button.hidden, true);
state.config.memory.pipelines = true;

// FR-5: no pipeline for the pair shows the create command.
responses[GET] = [200, { pipeline: null, runs: [] }];
await loadPipeline();
assert.match(body(), /No pipeline for this store pair\. Create one:/);
assert.match(body(),
  /agentbricks memory pipeline create --memory-store store --session-store sessions/);
assert.equal(button.hidden, true);

// FR-1, FR-2: settings and recent runs with state, start time, and error.
responses[GET] = [200, { pipeline, runs: [
  { name: 'memory-pipelines/p-1/runs/2', state: 'FAILED', create_time: '2026-10-20T09:14:00Z',
    error: 'Model service not found' },
  { name: 'memory-pipelines/p-1/runs/1', state: 'SUCCEEDED', create_time: '2026-10-19T15:02:00Z' },
] }];
await loadPipeline();
for (const text of ['dreaming-agent-memory · Manual only', 'system.ai.claude-sonnet-4-5',
  'Keep only durable preferences.', 'Failed', 'Succeeded', 'Model service not found',
  'Oct 20, 2026']) {
  assert.ok(body().includes(text), `missing ${text}: ${body()}`);
}
assert.equal(button.hidden, false);
assert.equal(button.disabled, false);
assert.equal(button.textContent, 'Run now');
assert.deepEqual(timers, [], 'no polling once runs are finished');
assert.equal(events.at(-1).type, 'memory.pipeline.get');

// FR-3: "Run now" opens the warning without starting a run; Cancel closes it.
requests.length = 0;
setPipelineConfirm(true);
assert.equal(confirmBox.hidden, false);
assert.match(elements.memoryPipelineConfirmText.textContent,
  /^Run the pipeline now\? It can rewrite, merge, or delete any memory in this store, including ones the agent saved\.$/);
assert.equal(button.disabled, true);
setPipelineConfirm(false);
assert.equal(confirmBox.hidden, true);
assert.equal(button.disabled, false);
assert.equal(requests.length, 0);

// FR-3, FR-4: confirming posts the run, then the running state disables the button and polls.
setPipelineConfirm(true);
log.length = 0;
responses[POST] = [200, { name: 'memory-pipelines/p-1/runs/3', state: 'RUNNING' }];
responses[GET] = [200, { pipeline, runs: [
  { name: 'memory-pipelines/p-1/runs/3', state: 'RUNNING', create_time: '2026-10-22T10:40:00Z' },
] }];
await runPipeline();
assert.deepEqual(requests, [POST, GET]);
assert.equal(confirmBox.hidden, true);
assert.ok(events.some((event) => event.type === 'memory.pipeline.run'));
assert.equal(button.disabled, true);
assert.equal(button.textContent, 'Running…');
assert.match(body(), /Running/);
assert.deepEqual(timers, [10000]);

// A closed pane stops polling.
elements.memoryPane.hidden = true;
await loadPipeline();
assert.deepEqual(timers, [10000]);
elements.memoryPane.hidden = false;

// A run already in progress shows the 409 detail inline.
responses[POST] = [409, { detail: 'A run is already in progress.',
  active_run: 'memory-pipelines/p-1/runs/3' }];
await runPipeline();
assert.match(body(), /A run is already in progress\./);
assert.equal(events.at(-2).type, 'memory.pipeline.error');
"""


def _js_source(framework):
    source = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "app.js").read_text()
    constants = re.search(
        r"^const PIPELINE_TRIGGERS = .*?^const pipelineState = .*?;$", source, re.M | re.S
    )
    assert constants, "Missing memory pipeline constants"
    functions = [constants.group()]
    for name in _JS_FUNCTIONS:
        match = re.search(rf"^(?:async )?function {name}\(.*?^\}}", source, re.M | re.S)
        assert match, f"Missing shipped UI function: {name}"
        functions.append(match.group())
    return "\n".join(functions)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_memory_pane_pipeline_block(framework):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    program = _js_source(framework) + _JS_HARNESS
    result = subprocess.run(
        [node, "--input-type=module", "-e", program, json.dumps(PIPELINE)],
        capture_output=True,
        text=True,
        timeout=10,
        env={"TZ": "UTC", "PATH": ""},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_run_now_opens_the_warning_and_only_its_run_button_starts_a_run(framework):
    source = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "app.js").read_text()

    assert "window.confirm" not in source
    for wiring in (
        'memoryPipelineRun.addEventListener("click", () => setPipelineConfirm(true));',
        'memoryPipelineCancel.addEventListener("click", () => setPipelineConfirm(false));',
        'memoryPipelineStart.addEventListener("click", () => void runPipeline());',
    ):
        assert wiring in source


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_memory_pane_has_the_pipeline_block(framework):
    html = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "index.html").read_text()

    assert 'id="memory-pipeline"' in html
    assert (
        '<button id="memory-pipeline-run" class="text-button" type="button" hidden>Run now</button>'
        in html
    )
    assert re.search(
        r'<div id="memory-pipeline-confirm" [^>]*role="alertdialog"[^>]* hidden>', html
    )
    assert (
        '<button id="memory-pipeline-start" class="text-button" type="button">Run</button>' in html
    )
    # The block sits between the store header and the Actor ID field.
    assert (
        html.index('id="memory-store-name"')
        < html.index('id="memory-pipeline"')
        < html.index('for="memory-actor-input"')
    )


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_hidden_run_now_button_is_not_displayed(framework):
    css = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "styles.css").read_text()

    # `.text-button` sets `display: inline-flex`, which overrides the `hidden` attribute.
    assert re.search(
        r"^\.memory-pipeline-head \.text-button\[hidden\] \{\s*display: none;\s*\}", css, re.M
    )
    assert re.search(r"^\.memory-pipeline-confirm\[hidden\] \{\s*display: none;\s*\}", css, re.M)
