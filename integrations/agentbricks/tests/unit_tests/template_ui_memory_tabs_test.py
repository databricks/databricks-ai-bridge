"""Check the chat app's Memory pane tabs (Entries | Pipeline) in both shipped templates."""

import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

UI_TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"
FRAMEWORKS = ["openai", "langgraph"]
VOID = {"input", "br", "img", "meta", "link", "hr", "path", "circle", "rect", "ellipse"}


class _PaneParser(HTMLParser):
    """Record each element's attributes and the ids of the elements it sits inside."""

    def __init__(self):
        super().__init__()
        self.elements = {}
        self.tabs = []
        self._stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        ancestors = [element_id for element_id in self._stack if element_id]
        if attrs.get("id"):
            self.elements[attrs["id"]] = {**attrs, "tag": tag, "ancestors": ancestors}
        if attrs.get("role") == "tab":
            self.tabs.append(attrs)
        if tag not in VOID:
            self._stack.append(attrs.get("id"))

    def handle_endtag(self, tag):
        if tag not in VOID and self._stack:
            self._stack.pop()


def _pane(framework):
    parser = _PaneParser()
    parser.feed((UI_TEMPLATES / f"agent-{framework}" / "ui" / "index.html").read_text())
    return parser


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_tabs_point_at_their_panels(framework):
    pane = _pane(framework)
    assert [tab["id"] for tab in pane.tabs] == ["memory-tab-entries", "memory-tab-pipeline"]
    for tab, panel_id in zip(pane.tabs, ["memory-entries-panel", "memory-pipeline"], strict=True):
        panel = pane.elements[panel_id]
        assert tab["aria-controls"] == panel_id
        assert panel["role"] == "tabpanel"
        assert panel["aria-labelledby"] == tab["id"]
    entries_tab, pipeline_tab = pane.tabs
    # Entries is selected on load; the Pipeline tab and panel start out of the tab order and hidden.
    assert entries_tab["aria-selected"] == "true"
    assert pipeline_tab["aria-selected"] == "false"
    assert pipeline_tab["tabindex"] == "-1"
    assert "hidden" not in pane.elements["memory-entries-panel"]
    assert "hidden" in pane.elements["memory-pipeline"]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_tabs_hide_with_the_browse_view(framework):
    pane = _pane(framework)
    # The empty states hide #memory-browse, so the tabs and both panels must sit inside it.
    for element_id in ("memory-tab-entries", "memory-entries-panel", "memory-pipeline"):
        assert "memory-browse" in pane.elements[element_id]["ancestors"], element_id
    assert "memory-browse" not in pane.elements["memory-setup"]["ancestors"]
    # The actor field, search and entries list belong to the Entries tab.
    for element_id in ("memory-actor-input", "memory-search-input", "memory-entries"):
        assert "memory-entries-panel" in pane.elements[element_id]["ancestors"], element_id


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_hidden_tab_panels_are_not_displayed(framework):
    css = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "styles.css").read_text()
    assert re.search(r"^\.memory-tab-panel\[hidden\] \{\s*display: none;\s*\}", css, re.M)
    # An id selector would outrank `.memory-tab-panel[hidden]` and keep an inactive panel showing.
    assert not re.search(r"^#memory-(entries-panel|pipeline)\s*[{,:\[]", css, re.M)


_TAB_HARNESS = r"""
import assert from 'node:assert/strict';
const node = () => ({ attrs: {}, hidden: false, tabIndex: 0, focused: false,
  classList: { names: new Set(), toggle(name, on) { on ? this.names.add(name) : this.names.delete(name); } },
  setAttribute(key, value) { this.attrs[key] = value; }, focus() { this.focused = true; } });
const elements = { memoryTabEntries: node(), memoryTabPipeline: node(),
  memoryEntriesPanel: node(), memoryPipeline: node(), memoryPipelineAlert: node() };
const confirms = [];
const setPipelineConfirm = (open) => confirms.push(open);
const { memoryTabEntries: entriesTab, memoryTabPipeline: pipelineTab,
  memoryEntriesPanel: entriesPanel, memoryPipeline: pipelinePanel } = elements;

selectMemoryTab('pipeline');
assert.equal(entriesPanel.hidden, true);
assert.equal(pipelinePanel.hidden, false);
assert.equal(pipelineTab.attrs['aria-selected'], 'true');
assert.equal(entriesTab.attrs['aria-selected'], 'false');
assert.equal(pipelineTab.tabIndex, 0);
assert.equal(entriesTab.tabIndex, -1);
assert.ok(pipelineTab.classList.names.has('active'));
assert.equal(pipelineTab.focused, false, 'a click selects without moving focus');
assert.deepEqual(confirms, [], 'opening Pipeline leaves the Run now warning alone');

// Leaving Pipeline closes a half-confirmed Run now warning.
selectMemoryTab('entries', { focus: true });
assert.equal(entriesPanel.hidden, false);
assert.equal(pipelinePanel.hidden, true);
assert.equal(entriesTab.focused, true);
assert.ok(!pipelineTab.classList.names.has('active'));
assert.deepEqual(confirms, [false]);

setPipelineAlert(true);
assert.equal(elements.memoryPipelineAlert.hidden, false);
setPipelineAlert(false);
assert.equal(elements.memoryPipelineAlert.hidden, true);
"""


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_select_memory_tab_switches_panels(framework):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    source = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "app.js").read_text()
    functions = []
    for name in ("selectMemoryTab", "setPipelineAlert"):
        match = re.search(rf"^function {name}\(.*?^\}}", source, re.M | re.S)
        assert match, f"Missing shipped UI function: {name}"
        functions.append(match.group())
    result = subprocess.run(
        [node, "--input-type=module", "-e", "\n".join(functions) + _TAB_HARNESS],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_tabs_are_wired_and_reset_on_open(framework):
    source = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "app.js").read_text()
    for wiring in (
        'memoryTabEntries.addEventListener("click", () => selectMemoryTab("entries"));',
        'memoryTabPipeline.addEventListener("click", () => selectMemoryTab("pipeline"));',
        '["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)',
    ):
        assert wiring in source
    open_pane = re.search(r"^function openMemoryPane\(.*?^\}", source, re.M | re.S)
    assert open_pane and 'selectMemoryTab("entries");' in open_pane.group()
