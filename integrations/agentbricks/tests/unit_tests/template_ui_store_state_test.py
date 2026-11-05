"""Check the chat app's store states in both shipped templates, by running the shipped app.js."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI_TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"
FRAMEWORKS = ["openai", "langgraph"]

_STUBS = r"""
import assert from 'node:assert/strict';
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.textContent = ''; this.hidden = false;
    this.className = ''; this.attrs = {}; this.firstChild = null; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener() {}
  set innerHTML(value) { this.firstChild = new Node('svg'); }
  text() {
    return [this.textContent, ...this.children.map((c) => (typeof c === 'string' ? c : c.text()))]
      .join(' ');
  }
}
const document = { createElement: (tag) => new Node(tag) };
const window = { setTimeout: () => 0, clearTimeout: () => {} };
const navigator = { clipboard: { writeText: async () => {} } };
const pipelineState = { timer: null, starting: false };
const elements = { memorySetup: new Node('div'), memoryBrowse: new Node('div'),
  memoryPipelineRun: new Node('button'), memoryPipelineBody: new Node('div'),
  memoryPipelineAlert: new Node('span') };
"""

_MEMORY_FUNCTIONS = (
    "iconSvg commandRow codeText memoryEmptyIcon placeholderCommandRow memoryStoreName "
    "renderMemorySetup loadMemory pipelineText setPipelineAlert loadPipeline"
).split()


def _functions(framework, names):
    source = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "app.js").read_text()
    functions = []
    for name in names:
        match = re.search(rf"^(?:async )?function {name}\(.*?^\}}", source, re.M | re.S)
        assert match, f"Missing shipped UI function: {name}"
        functions.append(match.group())
    return "\n".join(functions)


def _run(program):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    result = subprocess.run(
        [node, "--input-type=module", "-e", program],
        capture_output=True,
        text=True,
        timeout=10,
        env={"TZ": "UTC", "PATH": ""},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_config_starts_null(framework):
    # The Memory pane can open before /api/ui/config answers, or after it fails: config is null.
    source = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "app.js").read_text()
    assert re.search(r"^const state = \{\n(?:  .*\n)*?  config: null,$", source, re.M)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_memory_pane_renders_without_config(framework):
    program = (
        _STUBS
        + "const state = { config: null };\n"
        + _functions(framework, _MEMORY_FUNCTIONS)
        + r"""
assert.equal(memoryStoreName(), 'Memory store');
await loadMemory();
assert.equal(elements.memorySetup.hidden, false);
assert.equal(elements.memoryBrowse.hidden, true);
assert.match(elements.memorySetup.text(), /No memory store yet/);
await loadPipeline();
assert.equal(elements.memoryPipelineRun.hidden, true);
assert.match(elements.memoryPipelineBody.text(), /need a deployed app/);
"""
    )
    _run(program)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_session_card_says_a_bound_store_connects_on_deploy(framework):
    program = (
        _STUBS
        + _functions(framework, ["sessionCapability"])
        + r"""
const local = { history: true, managed: false, store: null, actor: 'agent', bound_store: null };
const config = (session, deployed = false) => ({ session: { ...local, ...session }, deployed });

// Nothing bound: keep asking to bind one.
let [orb, detail, command] = sessionCapability(config({}));
assert.equal(orb, 'degraded');
assert.match(detail, /kept in process and resets/);
assert.equal(command, 'agentbricks sessions bind <store-name>');

// Bound under `agentbricks dev`: don't ask to bind again; it connects on deploy.
[orb, detail, command] = sessionCapability(config({ bound_store: 'agent-sessions' }));
assert.equal(orb, 'degraded');
assert.match(detail, /Session store "agent-sessions" is bound/);
assert.match(detail, /until you deploy/);
assert.equal(command, 'agentbricks deploy');

// The bound-store note is dev-only, so a deployed app without a store falls back to bind.
[, , command] = sessionCapability(config({ bound_store: 'agent-sessions' }, true));
assert.equal(command, 'agentbricks sessions bind <store-name>');

// Connected: no command.
[orb, detail, command] = sessionCapability(
  config({ managed: true, store: 'agent-sessions', bound_store: 'agent-sessions' }, true),
);
assert.equal(orb, true);
assert.match(detail, /Connected to Session Store "agent-sessions" for actor agent/);
assert.equal(command, null);
"""
    )
    _run(program)
