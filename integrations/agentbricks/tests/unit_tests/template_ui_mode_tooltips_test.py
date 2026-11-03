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
def test_mode_tooltip_visibility_is_driven_by_one_class(template):
    css = (UI_ROOT / template / "ui/styles.css").read_text()
    reveals = re.findall(r"([^{}/]*)\{\s*visibility:\s*visible;\s*\}", css)
    # Stacked :hover/:focus reveal rules let two tips show at once, so JS owns visibility — for the
    # mode tips and the local-run viewer tip alike.
    assert [selector.strip() for selector in reveals] == [
        ".viewer-chip.tip-visible > .viewer-tip",
        ".mode-option.tip-visible > .mode-tip",
    ]
    assert ":hover > .mode-tip" not in css
    assert ":focus-visible + .mode-tip" not in css
    assert ":hover > .viewer-tip" not in css
    assert ":focus-visible > .viewer-tip" not in css


@pytest.mark.parametrize("template", TEMPLATES)
def test_at_most_one_mode_tooltip_shows(template):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    source = (UI_ROOT / template / "ui/app.js").read_text()
    match = re.search(
        r"^// At most one mode tip shows.*?"
        r'^document\.addEventListener\("keydown".*?^\}\);',
        source,
        re.M | re.S,
    )
    assert match, "Missing mode tip wiring in app.js"
    program = (
        r"""
import assert from 'node:assert/strict';
const target = () => ({ listeners: {},
  addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); },
  fire(type, event = {}) { for (const handler of this.listeners[type] ?? []) handler(event); } });
const classList = () => { const names = new Set(); return { names,
  add: name => names.add(name),
  contains: name => names.has(name),
  toggle: (name, on) => on ? names.add(name) : names.delete(name) }; };
const options = ['streaming', 'background', 'sync'].map(mode => {
  const button = Object.assign(target(), { dataset: { mode }, classList: classList(),
    focusVisible: false, matches: selector => selector === ':focus-visible' && button.focusVisible });
  return Object.assign(target(), { mode, button, classList: classList(), querySelector: () => button });
});
const document = Object.assign(target(), { querySelectorAll: selector =>
  selector === '.mode-option' ? options : options.map(option => option.button) });
const window = { matchMedia: () => ({ matches: false }) };
const state = {};
"""
        + match.group()
        + r"""
const [streaming, background, sync] = options;
const shown = () => options.filter(option => option.classList.contains('tip-visible')).map(option => option.mode);
const mouse = { pointerType: 'mouse' };

// Sweep the pointer across every button, including out-of-order leave events.
for (const option of options) {
  option.fire('pointerenter', mouse);
  assert.deepEqual(shown(), [option.mode]);
}
streaming.fire('pointerleave', mouse);
assert.deepEqual(shown(), ['sync'], 'a late leave from another option keeps the current tip');
sync.fire('pointerleave', mouse);
assert.deepEqual(shown(), []);

// Tab to Streaming, then hover Background without clicking (the reported overlap).
streaming.button.focusVisible = true;
streaming.button.fire('focus');
assert.deepEqual(shown(), ['streaming']);
background.fire('pointerenter', mouse);
assert.deepEqual(shown(), ['background']);
streaming.button.fire('blur');
assert.deepEqual(shown(), ['background'], 'blurring a hidden tip leaves the hovered one');
background.fire('pointerleave', mouse);
assert.deepEqual(shown(), []);

// A mouse click focuses without :focus-visible, so it shows nothing extra.
streaming.button.focusVisible = false;
sync.button.fire('focus');
assert.deepEqual(shown(), []);

// Touch taps don't flash a tip via pointer events.
background.fire('pointerenter', { pointerType: 'touch' });
assert.deepEqual(shown(), []);

// Escape hides whichever tip is up.
background.fire('pointerenter', mouse);
document.fire('keydown', { key: 'Enter' });
assert.deepEqual(shown(), ['background']);
document.fire('keydown', { key: 'Escape' });
assert.deepEqual(shown(), []);

background.button.fire('click');
assert.equal(state.mode, 'background', 'clicking still selects the mode');
assert.deepEqual(options.map(option => option.button.classList.contains('active')), [false, true, false]);
"""
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", program],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
