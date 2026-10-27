"""Exercise how the shipped chat UIs report a failed invocation, with only transport stubbed."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"


@pytest.mark.parametrize("template", ["agent-openai", "agent-langgraph"])
def test_failed_turn_shows_agent_error_message(template):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    source = (TEMPLATES / template / "ui/app.js").read_text()
    functions = []
    for name in "errorMessage jsonResponse handleEvent parseSseFrame invokeStreaming".split():
        match = re.search(rf"^(?:async )?function {name}\(.*?^\}}", source, re.M | re.S)
        assert match, f"Missing shipped UI function: {name}"
        functions.append(match.group())
    program = (
        "\n".join(functions)
        + r"""
import assert from 'node:assert/strict';
const state = {};
const addEvent = () => {};
const finishDraft = () => {};
const invocationPayload = () => ({ id: 'run-1' });
const invocationHeaders = () => ({});
const error = { code: 'UPSTREAM_RATE_LIMITED', message: 'Exceeded workspace rate limit' };
const json = (status, body) => ({ ok: status < 400, status, json: async () => body });

// Synchronous: the failed invocation body carries an error object.
await assert.rejects(
  jsonResponse(json(429, { id: 'run-1', status: 'failed', error })),
  { message: 'Agent invocation failed: Exceeded workspace rate limit' },
);
// Auth failures keep their {"error": {...}} body and show their own message.
await assert.rejects(
  jsonResponse(json(403, { error: { code: 'MCP_PERMISSION_DENIED', message: 'Permission denied' } })),
  { message: 'Permission denied' },
);
await assert.rejects(jsonResponse(json(404, { detail: 'invocation not found' })), { message: 'invocation not found' });

// Streaming: run.failed carries the same error object.
const stream = (events) => ({
  ok: true,
  body: { getReader: () => {
    const chunks = events.map((event) => new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`));
    return { read: async () => (chunks.length ? { value: chunks.shift(), done: false } : { done: true }) };
  } },
});
globalThis.fetch = async () => stream([{ type: 'run.started' }, { type: 'run.failed', error }]);
await assert.rejects(invokeStreaming({}), { message: 'Agent invocation failed: Exceeded workspace rate limit' });
globalThis.fetch = async () => stream([{ type: 'run.started' }, { type: 'run.failed' }]);
await assert.rejects(invokeStreaming({}), /Agent invocation run-1 failed/);
"""
    )
    result = subprocess.run(
        [node, "--input-type=module", "-"],
        input=program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
