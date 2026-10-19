"""Check the chat UI's invocation-mode tooltips in both shipped templates."""

import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

UI_ROOT = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"
TEMPLATES = ["agent-openai", "agent-langgraph"]
EXPECTED_TIPS = {
    "streaming": "Shows the answer as it's generated.",
    "background": "Runs on the server, even if you close the tab.",
    "sync": "Waits for the agent, then shows the full answer.",
}


class _ModeParser(HTMLParser):
    """Collect mode buttons and the text of every element with an id."""

    def __init__(self):
        super().__init__()
        self.buttons = []
        self.roles = {}
        self.texts = {}
        self._open = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "button" and "mode-button" in (attrs.get("class") or "").split():
            self.buttons.append(attrs)
        self._open.append(attrs.get("id"))
        if attrs.get("id"):
            self.roles[attrs["id"]] = attrs.get("role")
            self.texts[attrs["id"]] = ""

    def handle_endtag(self, tag):
        if self._open:
            self._open.pop()

    def handle_data(self, data):
        for element_id in filter(None, self._open):
            self.texts[element_id] += data


def _parse(template):
    parser = _ModeParser()
    parser.feed((UI_ROOT / template / "ui/index.html").read_text())
    return parser


def _tips(template):
    parser = _parse(template)
    return {
        button["data-mode"]: parser.texts[button["aria-describedby"]].strip()
        for button in parser.buttons
    }


@pytest.mark.parametrize("template", TEMPLATES)
def test_mode_buttons_are_described_by_tooltips(template):
    parser = _parse(template)
    assert [button["data-mode"] for button in parser.buttons] == list(EXPECTED_TIPS)
    for button in parser.buttons:
        assert "title" not in button, "native title tooltips double up with the custom tip"
        tip_id = button.get("aria-describedby")
        assert tip_id, f"{button['data-mode']} button has no aria-describedby"
        assert parser.roles.get(tip_id) == "tooltip"


@pytest.mark.parametrize("template", TEMPLATES)
def test_mode_tooltip_copy_is_one_short_line(template):
    tips = _tips(template)
    assert tips == EXPECTED_TIPS
    for text in tips.values():
        assert "\n" not in text and len(text) <= 60


def test_mode_tooltips_match_across_templates():
    assert _tips("agent-openai") == _tips("agent-langgraph")
    styles = [(UI_ROOT / name / "ui/styles.css").read_text() for name in TEMPLATES]
    assert styles[0] == styles[1]


@pytest.mark.parametrize("template", TEMPLATES)
def test_mode_tooltip_shows_on_hover_and_focus(template):
    css = (UI_ROOT / template / "ui/styles.css").read_text()
    reveal = re.search(r"([^{}]*)\{\s*visibility:\s*visible;\s*\}", css)
    assert reveal, "no rule reveals the mode tooltip"
    selectors = [selector.strip() for selector in reveal.group(1).split(",")]
    assert ".mode-option:hover > .mode-tip" in selectors
    assert ".mode-button:focus-visible + .mode-tip" in selectors
    assert re.search(
        r"\.mode-button\.tip-dismissed \+ \.mode-tip\s*\{\s*visibility:\s*hidden;", css
    )
    # The dismissed rule must come after the reveal rules so Escape wins.
    assert css.index(".mode-button.tip-dismissed") > reveal.start()


@pytest.mark.parametrize("template", TEMPLATES)
def test_escape_dismisses_mode_tooltip_until_focus_or_pointer_leaves(template):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    source = (UI_ROOT / template / "ui/app.js").read_text()
    match = re.search(
        r'^document\.querySelectorAll\("\.mode-button"\)\.forEach\(.*?'
        r"^// Escape hides the mode tip.*?^\}\);",
        source,
        re.M | re.S,
    )
    assert match, "Missing mode-button wiring in app.js"
    program = (
        r"""
import assert from 'node:assert/strict';
const target = () => ({ listeners: {},
  addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); },
  fire(type, event = {}) { for (const handler of this.listeners[type] ?? []) handler(event); } });
const button = Object.assign(target(), { dataset: { mode: 'background' }, classes: new Set(),
  parentElement: target(), hovered: false, focused: false });
button.classList = {
  add: name => button.classes.add(name),
  remove: name => button.classes.delete(name),
  toggle: (name, on) => on ? button.classes.add(name) : button.classes.delete(name),
};
// Only the Escape handler's hover/focus query depends on the button's state.
const document = Object.assign(target(), { querySelectorAll: selector =>
  selector === '.mode-button' || button.hovered || button.focused ? [button] : [] });
const state = {};
"""
        + match.group()
        + r"""
const dismissed = () => button.classes.has('tip-dismissed');
document.fire('keydown', { key: 'Escape' });
assert.equal(dismissed(), false, 'Escape ignores tips that are not shown');
button.focused = true;
document.fire('keydown', { key: 'Enter' });
assert.equal(dismissed(), false);
document.fire('keydown', { key: 'Escape' });
assert.equal(dismissed(), true, 'Escape hides the focused tip');
button.focused = false;
button.fire('blur');
assert.equal(dismissed(), false);
button.hovered = true;
document.fire('keydown', { key: 'Escape' });
assert.equal(dismissed(), true, 'Escape hides the hovered tip');
button.parentElement.fire('mouseleave');
assert.equal(dismissed(), false);
button.fire('click');
assert.equal(state.mode, 'background', 'clicking still selects the mode');
"""
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", program],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
