"""The chat app's Tools row and pane: the /api/demo/tools route and the shipped rendering logic."""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import types
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from databricks_agentbricks.cli.tools import tools as tools_group
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.projects.config import write_project_metadata

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"
FRAMEWORKS = ["openai", "langgraph"]

EXPECTED_ADD_COMMANDS = [
    {
        "kind": "mcp",
        "label": "Managed MCP service",
        "command": "agentbricks tools add mcp system.ai.web_search",
    },
    {
        "kind": "uc-function",
        "label": "Unity Catalog function",
        "command": "agentbricks tools add uc-function catalog.schema.function",
    },
    {"kind": "genie-one", "label": "Genie One", "command": "agentbricks tools add genie-one"},
    {
        "kind": "genie-agent",
        "label": "Genie space",
        "command": "agentbricks tools add genie-agent SPACE_ID",
    },
    {
        "kind": "sandbox",
        "label": "Data sandbox",
        "command": "agentbricks tools add sandbox --scope volume:catalog.schema.volume",
    },
    {
        "kind": "discover",
        "label": "Find more MCP services",
        "command": "agentbricks tools list --kind mcp",
    },
]

SPACE_ID = "0123456789abcdef0123456789abcdef"
ALL_KINDS_TOOLS = f"""
[[tools]]
id = "web_search"
auth = "user"
source = {{ kind = "mcp", service = "system.ai.web_search" }}

[[tools]]
id = "lookup"
source = {{ kind = "uc_function", function = "main.tools.lookup" }}

[[tools]]
id = "genie_one"
auth = "user"
source = {{ kind = "genie_one" }}

[[tools]]
id = "sales"
auth = "app"
source = {{ kind = "genie_agent", space_id = "{SPACE_ID}" }}

[[tools]]
id = "sandbox"
auth = "user"
source = {{ kind = "sandbox", service = "system.ai.sandbox" }}
policy = {{ downscope = [
  {{ resource = "volume:main.data.files", permission = "read_only" }},
  {{ resource = "volume:main.data.more", permission = "read_write" }},
] }}
"""


class _Tool:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


def _sample_tools():
    return [
        _Tool("get_current_time", "Get the current date and time."),
        _Tool("send_message", "Send a message to a recipient.\n\nArgs:\n    to: who to message."),
    ]


def _manifest(root: Path, framework: str, tools: str = "") -> Path:
    path = root / "agent.toml"
    path.write_text(
        f'schema_version = 1\n\n[agent]\nname = "demo"\nframework = "{framework}"\n{tools}'
    )
    return path


@pytest.fixture(params=FRAMEWORKS)
def overlay(request, tmp_path, monkeypatch):
    """One overlay's runtime/ui.py, bound to a temp project with a stubbed agent.tools."""
    framework = request.param
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(tmp_path))
    _manifest(tmp_path, framework)
    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = []
    agent_tools = types.ModuleType("agent.tools")
    agent_tools.all_tools = _sample_tools  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.tools", agent_tools)
    spec = importlib.util.spec_from_file_location(
        f"{framework}_tools_ui", TEMPLATES / f"ui/agent-{framework}/runtime/ui.py"
    )
    assert spec and spec.loader
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    app = FastAPI()
    ui.install_ui(app)
    return types.SimpleNamespace(
        framework=framework,
        root=tmp_path,
        ui=ui,
        agent_tools=agent_tools,
        client=TestClient(app),
    )


def _get_tools(overlay) -> dict:
    response = overlay.client.get("/api/demo/tools", params={"session_id": "s"})
    assert response.status_code == 200, response.text
    return response.json()


def test_connected_lists_every_binding_kind_in_manifest_order(overlay):
    _manifest(overlay.root, overlay.framework, ALL_KINDS_TOOLS)
    assert _get_tools(overlay)["connected"] == [
        {
            "id": "web_search",
            "kind": "mcp",
            "label": "MCP service",
            "source": "system.ai.web_search",
            "auth": "user",
        },
        {
            "id": "lookup",
            "kind": "uc_function",
            "label": "UC function",
            "source": "main.tools.lookup",
            "auth": "app",
        },
        {
            "id": "genie_one",
            "kind": "genie_one",
            "label": "Genie One",
            "source": "",
            "auth": "user",
        },
        {
            "id": "sales",
            "kind": "genie_agent",
            "label": "Genie space",
            "source": SPACE_ID,
            "auth": "app",
        },
        {
            "id": "sandbox",
            "kind": "sandbox",
            "label": "Data sandbox",
            "source": "volume:main.data.files, volume:main.data.more",
            "auth": "user",
        },
    ]


def test_connected_is_empty_without_bindings(overlay):
    payload = _get_tools(overlay)
    assert payload["connected"] == []
    assert payload["errors"] == []


def test_local_lists_python_tools_with_first_line_descriptions(overlay):
    assert _get_tools(overlay)["local"] == [
        {"name": "get_current_time", "description": "Get the current date and time."},
        {"name": "send_message", "description": "Send a message to a recipient."},
    ]


def test_add_commands_match_the_cli_recipes(overlay):
    assert _get_tools(overlay)["add_commands"] == EXPECTED_ADD_COMMANDS


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_overlay_add_commands_are_identical_and_pinned(framework):
    source = (TEMPLATES / f"ui/agent-{framework}/runtime/ui.py").read_text()
    namespace: dict = {}
    match = re.search(r"^_ADD_COMMANDS = \(.*?^\)", source, re.M | re.S)
    assert match
    exec(match.group(), namespace)  # noqa: S102 - evaluates a literal tuple from our own template
    assert list(namespace["_ADD_COMMANDS"]) == EXPECTED_ADD_COMMANDS


def _parse(args: list[str]) -> click.Context:
    """Resolve and parse a `tools …` command line with the real Click group, without running it."""
    command: click.Command = tools_group
    ctx = click.Context(command, info_name="tools")
    args = list(args)
    while isinstance(command, click.Group):
        name, *args = args
        sub = command.get_command(ctx, name)
        assert sub is not None, name
        command = sub
        ctx = click.Context(command, info_name=name, parent=ctx)
    return command.make_context(ctx.info_name, args, parent=ctx.parent)


@pytest.mark.parametrize("entry", EXPECTED_ADD_COMMANDS, ids=lambda entry: entry["kind"])
def test_every_add_command_parses_with_the_cli(entry):
    argv = shlex.split(entry["command"])
    assert argv[:2] == ["agentbricks", "tools"]
    ctx = _parse(argv[2:])
    assert ctx.command.name == ("list" if entry["kind"] == "discover" else entry["kind"])


def test_add_kinds_cover_the_cli_recipes():
    list_command = tools_group.commands["list"]
    kind_option = next(param for param in list_command.params if param.name == "kind")
    assert isinstance(kind_option.type, click.Choice)
    kinds = {entry["kind"] for entry in EXPECTED_ADD_COMMANDS if entry["kind"] != "discover"}
    assert kinds == set(kind_option.type.choices)
    assert set(tools_group.commands["add"].commands) == kinds  # ty: ignore[unresolved-attribute]

    class _Obj:
        output = "json"

    # The local recipes `tools list` prints are the exact commands the pane offers.
    for kind in ("sandbox", "uc-function", "genie-one", "genie-agent"):
        result = CliRunner().invoke(tools_group, ["list", "--kind", kind], obj=_Obj())
        assert result.exit_code == 0, result.output
        [recipe] = json.loads(result.output)["available_tools"]
        assert {"kind": kind, "command": recipe["add_command"]} in [
            {"kind": entry["kind"], "command": entry["command"]} for entry in EXPECTED_ADD_COMMANDS
        ]


def test_invalid_manifest_reports_an_error_and_keeps_other_sections(overlay):
    _manifest(
        overlay.root,
        overlay.framework,
        '\n[[tools]]\nid = "bad"\nsource = { kind = "nope" }\n',
    )
    payload = _get_tools(overlay)
    assert payload["connected"] == []
    assert [error["section"] for error in payload["errors"]] == ["connected"]
    assert "Unsupported agent.toml tool kind: 'nope'" in payload["errors"][0]["message"]
    assert [tool["name"] for tool in payload["local"]] == ["get_current_time", "send_message"]
    assert payload["add_commands"] == EXPECTED_ADD_COMMANDS


def test_unparseable_manifest_reports_an_error(overlay):
    (overlay.root / "agent.toml").write_text("schema_version = [\n")
    payload = _get_tools(overlay)
    assert payload["connected"] == []
    assert "Could not read" in payload["errors"][0]["message"]


def test_agent_tools_import_failure_reports_an_error(overlay, monkeypatch):
    _manifest(overlay.root, overlay.framework, ALL_KINDS_TOOLS)

    def broken():
        raise ImportError("No module named 'missing_dependency'")

    monkeypatch.setattr(overlay.agent_tools, "all_tools", broken)
    payload = _get_tools(overlay)
    assert payload["local"] == []
    assert len(payload["connected"]) == 5
    assert payload["errors"] == [
        {
            "section": "local",
            "message": "Could not load agent.tools: No module named 'missing_dependency'",
        }
    ]


class _Ctx:
    output = "text"

    def client(self):
        class _Client:
            def get_mcp_service(self, service):
                return {"name": f"mcp-services/{service}"}

        return _Client()


def test_cli_add_and_remove_show_up_without_restarting(overlay):
    project = overlay.root
    (project / "agent.toml").unlink()
    write_project_metadata(
        project, framework=overlay.framework, template=f"agent-{overlay.framework}"
    )
    AgentProject.create(project, framework=overlay.framework, server="agentbricks").write()
    assert _get_tools(overlay)["connected"] == []

    added = CliRunner().invoke(
        tools_group,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output
    assert [(tool["id"], tool["source"]) for tool in _get_tools(overlay)["connected"]] == [
        ("web_search", "system.ai.web_search")
    ]

    removed = CliRunner().invoke(
        tools_group, ["remove", "web_search", "--source", str(project)], obj=_Ctx()
    )
    assert removed.exit_code == 0, removed.output
    assert _get_tools(overlay)["connected"] == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_tools_row_sits_between_memory_and_traces_and_pane_exists(framework):
    html = (TEMPLATES / f"ui/agent-{framework}/ui/index.html").read_text()
    order = re.findall(r'data-capability="(\w+)"', html)
    assert order == ["streaming", "background", "session", "memory", "tools", "traces"]
    assert 'id="tools-mode-value"' in html
    assert 'id="tools-status"' in html
    for element in ("tools-pane", "tools-back", "tools-refresh", "tools-body"):
        assert f'id="{element}"' in html


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_refresh_buttons_refetch_tools(framework):
    source = (TEMPLATES / f"ui/agent-{framework}/ui/app.js").read_text()
    load_config = re.search(r"^async function loadConfig\(.*?^\}", source, re.M | re.S)
    assert load_config and "void loadTools();" in load_config.group()
    assert 'fetch(demoUrl("/api/demo/tools"), { cache: "no-store" })' in source
    assert 'elements.refreshConfig.addEventListener("click", () => loadConfig()' in source
    assert 'elements.toolsRefresh.addEventListener("click", () => loadTools());' in source
    assert 'elements.toolsBack.addEventListener("click", closeToolsPane);' in source
    assert 'item?.dataset.capability === "tools"' in source


_RENDER_PROGRAM = r"""
import assert from 'node:assert/strict';

class Node {
  constructor(tag) {
    this.tag = tag; this.children = []; this.className = ''; this.attrs = {}; this.listeners = {};
    this._text = null; this.hidden = false;
    const classes = new Set();
    this.classList = {
      toggle: (name, on) => (on ? classes.add(name) : classes.delete(name), on),
      contains: (name) => classes.has(name),
      add: (name) => classes.add(name),
    };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() {
    if (this._text !== null) return this._text;
    return this.children.map((child) => (typeof child === 'string' ? child : child.textContent)).join('');
  }
  set innerHTML(value) { this._html = value; }
  append(...nodes) { this._text = null; this.children.push(...nodes); }
  replaceChildren(...nodes) { this._text = null; this.children = [...nodes]; }
  setAttribute(name, value) { this.attrs[name] = value; }
  addEventListener(type, handler) { this.listeners[type] = handler; }
  closest() { return null; }
  findAll(className) {
    const found = [];
    for (const child of this.children) {
      if (typeof child === 'string') continue;
      if (child.className.split(' ').includes(className)) found.push(child);
      found.push(...child.findAll(className));
    }
    return found;
  }
}
const document = { createElement: (tag) => new Node(tag) };
const copied = [];
const navigator = { clipboard: { writeText: async (text) => { copied.push(text); } } };
const elements = { toolsMode: new Node('small'), toolsStatus: new Node('span'), toolsBody: new Node('div') };

const addCommands = JSON.parse(process.argv[1]);
const payload = (connected, errors = []) => ({
  connected,
  local: [
    { name: 'get_current_time', description: 'Get the current date and time.' },
    { name: 'send_message', description: 'Send a message to a recipient.' },
  ],
  add_commands: addCommands,
  errors,
});
const texts = (className) => elements.toolsBody.findAll(className).map((node) => node.textContent);

// No managed bindings: amber orb, "None connected", the empty state, and the sample Python tools.
renderTools(payload([]));
assert.equal(elements.toolsMode.textContent, 'None connected · 2 in code');
assert.equal(elements.toolsStatus.classList.contains('degraded'), true);
assert.equal(elements.toolsStatus.classList.contains('enabled'), false);
assert.deepEqual(texts('tools-heading'), ['Connected', 'Defined in code', 'Add a tool']);
assert.deepEqual(texts('tools-empty'), ['No managed tools connected yet.']);
assert.deepEqual(texts('tool-badge'), []);
assert.deepEqual(texts('tool-name'), ['get_current_time', 'send_message']);
assert.deepEqual(texts('tools-hint'), [
  'Run in your project directory, then restart agentbricks dev or run agentbricks deploy.',
]);
assert.deepEqual(elements.toolsBody.findAll('tools-hint')[0].children.filter((c) => typeof c !== 'string')
  .map((c) => [c.tag, c.textContent]), [['code', 'agentbricks dev'], ['code', 'agentbricks deploy']]);

// Every add command is a labeled code row whose copy button copies the exact command.
assert.deepEqual(texts('cap-command-label'), addCommands.map((entry) => entry.label));
const rows = elements.toolsBody.findAll('cap-command');
assert.deepEqual(rows.map((row) => row.children[0].textContent), addCommands.map((entry) => entry.command));
for (const row of rows) {
  let stopped = false;
  row.children[1].listeners.click({ stopPropagation: () => { stopped = true; } });
  assert.equal(stopped, true);
}
await Promise.resolve();
assert.deepEqual(copied, addCommands.map((entry) => entry.command));

// One binding: green orb, a badge row per tool, and no empty state.
renderTools(payload([
  { id: 'web_search', kind: 'mcp', label: 'MCP service', source: 'system.ai.web_search', auth: 'user' },
  { id: 'genie_one', kind: 'genie_one', label: 'Genie One', source: '', auth: 'user' },
]));
assert.equal(elements.toolsMode.textContent, '2 connected · 2 in code');
assert.equal(elements.toolsStatus.classList.contains('enabled'), true);
assert.equal(elements.toolsStatus.classList.contains('degraded'), false);
assert.deepEqual(texts('tools-empty'), []);
assert.deepEqual(texts('tool-badge'), ['MCP service', 'Genie One']);
assert.deepEqual(texts('tool-name').slice(0, 2), ['web_search', 'genie_one']);
assert.deepEqual(texts('tool-detail').slice(0, 1), ['system.ai.web_search']);

// Errors render in the error state style ahead of the sections.
renderTools(payload([], [{ section: 'connected', message: 'Unsupported agent.toml tool kind' }]));
assert.equal(elements.toolsBody.children[0].className, 'state-error');
assert.equal(elements.toolsBody.children[0].textContent, 'Unsupported agent.toml tool kind');
assert.deepEqual(texts('tools-empty'), ['No managed tools connected yet.']);
"""


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_render_tools_row_and_pane(framework):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    source = (TEMPLATES / f"ui/agent-{framework}/ui/app.js").read_text()
    functions = []
    for name in ("setCapability", "commandRow", "toolsSection", "toolRow", "renderTools"):
        match = re.search(rf"^function {name}\(.*?^\}}", source, re.M | re.S)
        assert match, f"Missing shipped UI function: {name}"
        functions.append(match.group())
    constants = re.search(r"^const CAPABILITY_SUMMARY = .*?;$", source, re.M)
    assert constants
    program = constants.group() + "\n" + "\n".join(functions) + _RENDER_PROGRAM
    result = subprocess.run(
        [node, "--input-type=module", "-e", program, json.dumps(EXPECTED_ADD_COMMANDS)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
