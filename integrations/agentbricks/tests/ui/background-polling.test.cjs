const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

function polling(framework, fetch) {
  const source = fs.readFileSync(path.resolve(__dirname, "../../src/databricks_agentbricks/templates/ui", framework, "ui/app.js"), "utf8");
  // Exercise the production polling functions with controlled HTTP responses and timers.
  // Full page/session switching behavior is covered by tests/e2e/chat_browser.py.
  const code = source.slice(source.indexOf("function waitForBackgroundPoll("), source.indexOf("async function invokeBackground("));
  const state = { backgroundRun: "run-1" };
  const button = { hidden: false };
  const output = [];
  const context = {
    state, AbortController, fetch,
    setTimeout: (callback) => setTimeout(callback, 0), clearTimeout,
    document: { querySelector: () => button },
    routingHeaders: () => ({ "X-Routing-Key": "session-1" }),
    jsonResponse: async (response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    },
    errorText: String, addEvent: () => {}, setStatus: () => {},
    agentResult: (result) => result.output,
    handleOutput: (items) => output.push(...items),
  };
  vm.createContext(context);
  vm.runInContext(code, context);
  return { context, state, button, output, run: () => context.pollBackground("run-1") };
}

for (const framework of ["agent-openai", "agent-langgraph"]) {
  test(`${framework}: a missing run unlocks the composer instead of trapping the session`, async () => {
    const probe = polling(framework, async () => ({ ok: false, status: 404 }));
    await assert.rejects(probe.run(), /no longer available.*outcome cannot be recovered/);
    assert.equal(probe.state.backgroundRun, null);
    assert.equal(probe.state.backgroundWaiting, false);
    assert.equal(probe.button.hidden, true);
  });

  test(`${framework}: transient status errors retain the run for a safe retry`, async () => {
    const probe = polling(framework, async () => ({ ok: false, status: 503 }));
    await assert.rejects(probe.run(), /status is unknown; use Check result/);
    assert.equal(probe.state.backgroundRun, "run-1");
    assert.equal(probe.state.backgroundWaiting, false);
    assert.equal(probe.button.hidden, false);
  });

  test(`${framework}: pausing aborts a hanging status request without cancelling the run`, async () => {
    let began;
    const fetching = new Promise((resolve) => { began = resolve; });
    const probe = polling(framework, (_url, { signal }) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
      began();
    }));
    const result = probe.run();
    await fetching;
    probe.state.backgroundWaiting = false;
    probe.state.backgroundController.abort();
    assert.equal((await result).status, "waiting");
    assert.equal(probe.state.backgroundRun, "run-1");
    assert.equal(probe.state.backgroundController, null);
  });

  test(`${framework}: resumed polling renders the final result once`, async () => {
    const probe = polling(framework, async () => ({ ok: true, status: 200, json: async () => ({ status: "completed", output: { status: "completed", output: ["answer"] } }) }));
    assert.equal((await probe.run()).status, "completed");
    assert.deepEqual(probe.output, ["answer"]);
    assert.equal(probe.state.backgroundRun, null);
  });
}
