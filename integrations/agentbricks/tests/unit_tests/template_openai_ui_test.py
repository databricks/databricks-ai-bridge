"""Exercise the shipped approval UI logic with only rendering and transport stubbed."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = (
    Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui/agent-openai/ui/app.js"
)


@pytest.mark.parametrize("mode", ["sync", "streaming", "background"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_pending_approval_survives_chat_submission(mode, decision):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped JavaScript")
    source = APP_JS.read_text()
    names = (
        "setStatus updateSendState setBusy formatJson appendError finishDraft "
        "interruptSummary handleInterrupt dispatch sendText resume clearChat"
    ).split()
    functions = []
    for name in names:
        match = re.search(rf"^(?:async )?function {name}\(.*?^\}}", source, re.M | re.S)
        assert match, f"Missing shipped UI function: {name}"
        functions.append(match.group())
    program = (
        "\n".join(functions)
        + r"""
import assert from 'node:assert/strict';
const [mode, decision] = process.argv.slice(1);
const state = { busy: false, mode, config: { session: { history: true, durable: true } } };
const element = () => ({ value: 'another message', options: [], hidden: false,
  classList: { contains: () => false }, setAttribute() {}, replaceChildren() {}, focus() {} });
const elements = new Proxy({}, { get: (target, key) => target[key] ??= element() });
const document = { querySelectorAll: () => [] };
const messages = [], requests = [];
const appendMessage = (role, content) => messages.push({ role, content });
const addEvent = () => {};
const refreshSessionView = async () => {};
const transport = mode => async payload => {
  requests.push({ mode, payload });
  state.lastAssistantText = 'done';
};
const invokeSync = transport('sync');
const invokeStreaming = transport('streaming');
const invokeBackground = transport('background');
const pending = { value: { action_requests: [
  { name: 'first_tool', call_id: 'call-first', args: {} },
  { name: 'second_tool', call_id: 'call-second', args: {} },
] } };
handleInterrupt(pending);
setBusy(false);

await assert.rejects(sendText('another message', mode), /Approve or reject/);
assert.equal(requests.length, 0, 'ordinary send must not reach the agent');
assert.equal(state.pendingInterrupt, pending, 'keep the original approval and call IDs');
assert.equal(elements.approvalPanel.hidden, false);
assert.equal(elements.sendButton.disabled, true);
assert.equal(elements.approveAction.disabled, false);
assert.equal(elements.rejectAction.disabled, false);
assert.equal(messages.some(message => message.role === 'user'), false);
assert.match(messages.at(-1).content, /Approve or reject/);

await resume(decision);
assert.equal(requests.length, 1);
assert.equal(requests[0].mode, 'streaming');
assert.deepEqual(requests[0].payload.resume.decisions, ['call-first', 'call-second'].map(
  call_id => ({ type: decision, call_id,
    ...(decision === 'reject' ? { message: 'Rejected from the Agent Bricks demo UI.' } : {}) })
));
assert.equal(state.pendingInterrupt, null);
assert.equal(elements.approvalPanel.hidden, true);
assert.equal(elements.sendButton.disabled, false);
assert.equal(await sendText('follow-up', mode), 'done');
assert.deepEqual(requests[1], { mode, payload: { messages: [{ role: 'user', content: 'follow-up' }] } });

handleInterrupt(pending);
setBusy(false);
assert.equal(elements.sendButton.disabled, true);
clearChat();
assert.equal(state.pendingInterrupt, null);
assert.equal(elements.approvalPanel.hidden, true);
assert.equal(elements.sendButton.disabled, false, 'a new chat must allow ordinary messages');
"""
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", program, mode, decision],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
