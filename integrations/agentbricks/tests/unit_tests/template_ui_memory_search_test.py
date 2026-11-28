"""The chat app reads memory search hits from `results` and labels every pipeline trigger."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI_TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"
FRAMEWORKS = ["openai", "langgraph"]

_HARNESS = r"""
import assert from 'node:assert/strict';
const entry = (path) => ({ name: `memory-stores/s/entries/${path}`, path });

// `results` is the ranked list; the deprecated top-level array is ignored when both are present.
assert.deepEqual(
  memorySearchEntries({
    results: [{ managed_memory_entry: entry('a'), score: 0.9 }, { score: 0.1 }],
    managed_memory_entries: [entry('stale')],
  }),
  [entry('a')],
);
assert.deepEqual(memorySearchEntries({ results: [] }), []);
// A server without `results` still returns the deprecated array.
assert.deepEqual(memorySearchEntries({ managed_memory_entries: [entry('b')] }), [entry('b')]);
assert.deepEqual(memorySearchEntries(null), []);

assert.equal(PIPELINE_TRIGGERS.ALWAYS_ON, 'Always on');
assert.equal(PIPELINE_TRIGGERS.SCHEDULED, 'Every ~24h');
assert.equal(PIPELINE_TRIGGERS.MANUAL_ONLY, 'Manual only');
"""


def _source(framework: str) -> str:
    source = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "app.js").read_text()
    parts = []
    for name in ("memoryEntries", "memorySearchEntries"):
        match = re.search(rf"^function {name}\(.*?^\}}", source, re.M | re.S)
        assert match, f"Missing shipped UI function: {name}"
        parts.append(match.group())
    triggers = re.search(r"^const PIPELINE_TRIGGERS = .*?;$", source, re.M)
    assert triggers, "Missing PIPELINE_TRIGGERS"
    parts.append(triggers.group())
    return "\n".join(parts)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_memory_search_reads_results_and_triggers_are_labeled(framework):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    result = subprocess.run(
        [node, "--input-type=module", "-e", _source(framework) + _HARNESS],
        capture_output=True,
        text=True,
        timeout=10,
        env={"PATH": ""},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_search_uses_the_results_reader_and_listing_does_not(framework):
    source = (UI_TEMPLATES / f"agent-{framework}" / "ui" / "app.js").read_text()

    search = re.search(
        r'fetch\(demoUrl\("/api/demo/memory/search"\).*?entries = (\w+)\(', source, re.S
    )
    listing = re.search(r'fetch\(url, \{ cache: "no-store" \}\);\s*entries = (\w+)\(', source)

    assert search and search.group(1) == "memorySearchEntries"
    assert listing and listing.group(1) == "memoryEntries"
