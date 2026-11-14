"use strict";
// The shell, message renderer and composer follow the default agent-openai UI.
const $ = id => document.getElementById(id);
const elements = { chatLog: $("chat-log"), emptyState: $("empty-state") };
const ACTIVE = new Set(["queued", "running"]);
const API_TIMEOUTS = { stop: 210000, status: 15000 };
const state = {
  config: null, agents: [], agent: null, context: null, sessions: [], runs: [],
  preparation: null, preparationTimer: null, warmPool: null,
  pending: null, sandbox: null, busy: false, loading: false, recovery: null,
  source: null, pollTimer: null, messages: new Map(), historyUnavailable: false,
  stopOperations: new Map(), sandboxRequests: new Map(), sandboxes: new Map(),
  eventFilter: "all", eventSignature: null,
  model: "", models: [], modelsLoading: false,
};
function node(tag, text, className) {
  const value = document.createElement(tag);
  if (text !== undefined) value.textContent = text;
  if (className) value.className = className;
  return value;
}
function read(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function remember(key, value) {
  try { localStorage.setItem(key, value); } catch { /* Optional display preferences. */ }
}
function pendingKey(ctx) { return `sandbox-chat:pending:${ctx.agentId}:${ctx.sessionId}`; }
function stopKey(ctx) { return `sandbox-chat:stop:${ctx.agentId}:${ctx.sessionId}`; }
function savedObject(key) {
  try { return JSON.parse(read(key)); } catch { return null; }
}
function current(ctx) { return state.context === ctx && !ctx.controller.signal.aborted; }
function selected(ctx) { return state.context?.agentId === ctx.agentId && state.context?.sessionId === ctx.sessionId; }
function stopping(ctx = state.context) { return ctx && state.stopOperations.get(stopKey(ctx)); }
function nextSandboxRequest(ctx) {
  const key = stopKey(ctx);
  const request = (state.sandboxRequests.get(key) || 0) + 1;
  state.sandboxRequests.set(key, request);
  return request;
}
function contextUrl(ctx, path) {
  return `${path}?${new URLSearchParams({ agent_id: ctx.agentId, session_id: ctx.sessionId })}`;
}
async function api(path, body, ctx, timeoutMs = 0) {
  const controller = timeoutMs ? new AbortController() : ctx?.controller;
  const abort = () => controller.abort();
  if (timeoutMs && ctx) {
    ctx.controller.signal.addEventListener("abort", abort, { once: true });
    if (ctx.controller.signal.aborted) abort();
  }
  let timedOut = false;
  const timer = timeoutMs && setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetch(path, {
      cache: "no-store", ...(controller ? { signal: controller.signal } : {}),
      ...(body === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      }),
    });
    const text = await response.text();
    let data;
    try { data = JSON.parse(text); } catch {
      throw new Error(`HTTP ${response.status}: unexpected response. Check your sign-in, then reconnect.`);
    }
    if (!response.ok) {
      const error = new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
      error.status = response.status;
      throw error;
    }
    return data;
  } catch (error) {
    if (timedOut) throw new Error("Request timed out.");
    throw error;
  } finally {
    clearTimeout(timer);
    if (timeoutMs && ctx) ctx.controller.signal.removeEventListener("abort", abort);
  }
}
function notice(message = "", recovery = null) {
  $("request-notice").textContent = message;
  $("request-notice").hidden = !message;
  state.recovery = recovery;
  $("retry-run").hidden = !recovery;
  $("retry-run").textContent = recovery === "submit" ? "Reconnect this invocation" : recovery === "failed" ? "Retry this invocation" : "Reconnect";
  updateControls();
}
function supportsPreparation() { return state.config?.capabilities?.includes("session-preparation"); }
function preparing() { return ["submitting", "queued", "preparing", "unknown"].includes(state.preparation?.status); }
function needsPreparation() {
  return supportsPreparation() && (state.preparation?.status !== "ready"
    || !["running", "ready"].includes(state.sandbox?.status));
}
function initializingSession() {
  const row = latestRun();
  return state.loading || preparing() || (ACTIVE.has(row?.status)
    && Object.entries(row.lifecycle?.steps || {}).some(([step, value]) => step !== "agent_run" && value.status === "running"));
}
function updateControls() {
  const stop = stopping();
  const blocked = !state.agent || state.busy || state.loading || state.historyUnavailable || Boolean(state.pending) || Boolean(stop);
  $("send-button").disabled = blocked || needsPreparation() || !$("prompt-input").value.trim();
  $("model-select").disabled = blocked || state.modelsLoading || state.models.length === 0;
  $("prompt-input").disabled = !state.agent || state.loading;
  $("retry-run").disabled = Boolean(stop) || state.loading || (state.busy && state.recovery !== "monitor");
  const status = state.sandbox?.status;
  let reason = "";
  if (!state.agent) reason = "Choose an agent source first.";
  else if (stop) reason = "Waiting for shutdown confirmation. You can switch sessions while this finishes.";
  else if (preparing()) reason = "Session preparation is in progress.";
  else if (state.loading) reason = "Loading this session…";
  else if (state.pending) reason = "Confirm the pending invocation before stopping this session.";
  else if (state.busy || ACTIVE.has(latestRun()?.status)) reason = "An invocation is in progress. Stop is available after it finishes.";
  else if (state.historyUnavailable) reason = "Reconnect to load this session before stopping.";
  else if (["stopped", "terminated"].includes(status)) reason = "This session's sandbox is already stopped.";
  else if (status === "missing") reason = "This session's sandbox is missing.";
  else if (!state.sandbox?.sandbox_id || status === "not_created") reason = status === "unknown" ? "Reconnect to check this session's sandbox." : "Send a message to create this session's sandbox.";
  $("stop-session").disabled = Boolean(reason);
  $("stop-session").textContent = stop ? "Stopping…" : state.config?.sandbox_scope === "environment" ? "Stop shared sandbox" : "Stop this session";
  $("stop-session").title = reason;
  $("stop-reason").textContent = reason;
  $("stop-reason").hidden = !reason;
  const initializing = initializingSession();
  elements.chatLog.setAttribute("aria-busy", String(state.busy || initializing || Boolean(stop)));
  $("session-readiness").hidden = !initializing;
  $("readiness-message").textContent = state.loading ? "Checking session readiness…"
    : preparing() ? state.preparation.stage || "Preparing session…" : latestRun()?.stage || "Initializing agent…";
  const badge = $("sandbox-tab-badge");
  badge.textContent = stop ? "Stopping" : preparing() ? "Preparing" : state.pending ? "Pending" : state.busy || ACTIVE.has(latestRun()?.status) ? "Running" : "";
  badge.hidden = !badge.textContent;
  $("new-session").disabled = !state.agent;
  $("prepare-session").hidden = !supportsPreparation() || !needsPreparation();
  $("prepare-session").disabled = blocked || preparing();
  $("prepare-session").textContent = preparing() ? "Preparing session…" : state.preparation?.error ? "Retry preparation"
    : ["stopped", "terminated"].includes(status) ? "Resume session" : "Prepare session";
}
function setBusy(value) {
  state.busy = value;
  updateControls();
}

function renderModels() {
  const select = $("model-select");
  select.replaceChildren(node("option", "Agent default"));
  select.firstElementChild.value = "";
  for (const name of state.models) {
    const option = node("option", name.replace(/^system\.ai\./, ""));
    option.value = name;
    select.append(option);
  }
  select.value = state.model;
  updateControls();
}
async function loadModels() {
  if (state.modelsLoading) return;
  state.modelsLoading = true;
  $("model-notice").hidden = true;
  updateControls();
  try {
    const data = await api("/api/models", undefined, undefined, 15000);
    if (!Array.isArray(data.available)) throw new Error("Unexpected model list");
    state.models = [...new Set(data.available.filter(name => typeof name === "string" && name))];
    // Keep an already selected model if a later discovery response omits it.
    if (state.model && !state.models.includes(state.model)) state.models.unshift(state.model);
    renderModels();
    if (!state.models.length) throw new Error("No chat models listed");
  } catch {
    $("model-notice-text").textContent = "Couldn't load the model list. You can still send a message with the current selection.";
    $("model-notice").hidden = false;
  } finally {
    state.modelsLoading = false;
    updateControls();
  }
}
function hideEmptyState() { elements.emptyState.hidden = true; }

function formatClock(date = new Date()) {
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function messageName(role, label) {
  if (label) return label;
  if (role === "tool") return "Tool";
  if (role === "system") return "System";
  if (role === "error") return "Error";
  return "Agent";
}

function safeHttpUrl(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}
function externalLink(text, url, className) {
  const link = node("a", text, className);
  link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer";
  return link;
}
function renderCapabilities() {
  const workbench = state.config?.workbench;
  const sessions = workbench?.sessions;
  const tools = Array.isArray(workbench?.tools) ? workbench.tools.filter(name => typeof name === "string") : null;
  const tracing = workbench?.tracing;
  const capabilities = [
    ["streaming", "Run snapshots", true, "Text and execution updates stream from the selected invocation."],
    ["session", sessions?.mode === "managed" ? "Managed Session Store" : sessions?.mode === "sandbox-sqlite" ? "Sandbox SQLite" : sessions?.mode === "custom" ? "Custom handler" : "Run history", true,
      sessions?.mode === "managed" ? `Conversation state uses the managed store${sessions.store ? `: ${sessions.store}` : "."}`
        : sessions?.mode === "sandbox-sqlite" ? "Conversation state stays on this session's sandbox disk."
        : sessions?.mode === "custom" ? "Conversation storage depends on the custom handler."
        : "Runs are grouped by session. Agent conversation storage was not reported."],
    ["tools", tools ? tools.length ? `${tools.length} declared binding${tools.length === 1 ? "" : "s"}` : "No declared bindings" : "Not reported", Boolean(tools?.length),
      tools ? "Declared tool bindings are listed here. Local Python tools can also be called; completed calls appear in Events." : "Tool configuration was not reported by this environment."],
    ["tracing", tracing?.configured ? "Configured" : tracing ? "Not configured" : "Not reported", Boolean(tracing?.configured),
      tracing?.configured ? "Trace links appear with answers when the agent reports a trace." : tracing ? "Tracing is not configured for this agent." : "Tracing configuration was not reported by this environment."],
  ];
  for (const [key, label, enabled, description] of capabilities) {
    $(`${key}-mode-value`).textContent = label;
    $(`${key}-status`).className = `status-orb${enabled ? " enabled" : ""}`;
    const detail = $(`${key}-detail`);
    detail.replaceChildren(node("span", description));
    if (key === "tools" && tools?.length) {
      const list = node("ul");
      tools.forEach(name => list.append(node("li", name)));
      detail.append(list);
    }
    if (key === "tracing" && tracing?.configured) {
      if (tracing.experiment_name || tracing.experiment_id) detail.append(node("span", tracing.experiment_name || `Experiment ${tracing.experiment_id}`));
      const url = safeHttpUrl(tracing.url);
      if (url) detail.append(externalLink("Open experiment ↗", url));
    }
  }
}
function selectEvidenceTab(name, focus = false) {
  for (const key of ["events", "sandbox"]) {
    const selected = key === name;
    const tab = $(`${key}-tab`);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $(`${key}-panel`).hidden = !selected;
    if (selected && focus) tab.focus();
  }
}
function buildMessageActions(textEl) {
  const actions = document.createElement("div");
  actions.className = "message-actions";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "message-action";
  copy.innerHTML =
    '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span>Copy</span>';
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(textEl.dataset.source ?? textEl.textContent ?? "");
    } catch {
      /* clipboard may be unavailable */
    }
    const span = copy.querySelector("span");
    if (span) {
      span.textContent = "Copied";
      setTimeout(() => {
        span.textContent = "Copy";
      }, 1200);
    }
  });
  actions.append(copy);
  return actions;
}

const markdown = window.markdownit({ html: false }).disable("image");

function renderMessageText(element, content) {
  element.dataset.source = content;
  element.innerHTML = markdown.render(content);
  element.querySelectorAll("a").forEach(anchor => {
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
  });
}

function appendMessage(role, content, label, { time } = {}) {
  hideEmptyState();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  const text = document.createElement("div");
  text.className = "message-content";
  if (role === "assistant") renderMessageText(text, content);
  else text.textContent = content;

  if (role === "user") {
    wrapper.append(text);
  } else {
    const head = document.createElement("div");
    head.className = "message-head";
    const name = document.createElement("span");
    name.className = "message-name";
    name.textContent = messageName(role, label);
    const stamp = document.createElement("span");
    stamp.className = "message-time";
    stamp.textContent = time || formatClock();
    head.append(name, stamp);
    wrapper.append(head, text);
    if (role === "assistant") wrapper.append(buildMessageActions(text));
  }

  elements.chatLog.append(wrapper);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
  return { wrapper, text };
}

function inputText(input) {
  return typeof input?.message === "string" ? input.message : JSON.stringify(input ?? {}, null, 2);
}
function runText(row) {
  if (typeof row.display_text === "string") return row.display_text;
  const result = row.result?.result;
  if (typeof result?.reply === "string") return result.reply;
  return result != null ? JSON.stringify(result, null, 2) : row.stream?.text || "";
}
function cancelTextAnimation(parts) {
  if (parts.animation) cancelAnimationFrame(parts.animation);
  parts.animation = null;
}
function renderAssistantText(parts, text, animate) {
  const previousTarget = parts.targetText;
  parts.targetText = text;
  const element = parts.assistant.text;
  const shown = element.dataset.source || "";
  if (!animate || !text.startsWith(shown) || document.hidden || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    cancelTextAnimation(parts);
    if (shown !== text) renderMessageText(element, text);
    return;
  }
  if (shown === text) return;
  // Smooth transport batches over at most 80 ms. Never build a token queue or
  // hold the final result behind an animation; terminal snapshots flush directly.
  if (previousTarget !== text) parts.textDeadline = performance.now() + 80;
  if (parts.animation) return;
  const frame = now => {
    parts.animation = null;
    if (!element.isConnected) return;
    const before = element.dataset.source || "";
    const target = parts.targetText;
    const remaining = parts.textDeadline - now;
    const count = remaining <= 0 ? target.length : before.length + Math.max(1, Math.ceil((target.length - before.length) * Math.min(1, 16 / remaining)));
    // Do not split a UTF-16 surrogate pair at the display boundary.
    const end = count < target.length && /[\uD800-\uDBFF]/.test(target[count - 1]) ? count + 1 : count;
    const nearBottom = elements.chatLog.scrollHeight - elements.chatLog.scrollTop - elements.chatLog.clientHeight < 100;
    renderMessageText(element, target.slice(0, end));
    if (nearBottom) elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
    if (end < target.length) parts.animation = requestAnimationFrame(frame);
  };
  parts.animation = requestAnimationFrame(frame);
}
function renderRun(row, animate = false) {
  let parts = state.messages.get(row.id);
  if (!parts) {
    const time = row.created_at ? formatClock(new Date(row.created_at)) : undefined;
    appendMessage("user", inputText(row.input), "You", { time });
    parts = {};
    state.messages.set(row.id, parts);
  }
  const text = runText(row);
  if (text || ACTIVE.has(row.status)) {
    if (!parts.assistant) parts.assistant = appendMessage("assistant", animate ? "" : text, "Agent");
    renderAssistantText(parts, text, animate && ACTIVE.has(row.status));
    parts.assistant.wrapper.classList.toggle("streaming", ACTIVE.has(row.status));
    parts.assistant.wrapper.setAttribute("aria-label", ACTIVE.has(row.status) ? "Agent response in progress" : "Agent response");
  }
  if (parts.assistant) {
    parts.assistant.wrapper.dataset.runId = row.id;
    const url = safeHttpUrl(row.result?.result?.trace?.url);
    const actions = parts.assistant.wrapper.querySelector(".message-actions");
    let trace = actions.querySelector(".message-trace");
    if (url) {
      if (!trace) { trace = externalLink("View trace ↗", url, "message-action message-trace"); actions.append(trace); }
      trace.href = url;
    } else trace?.remove();
  }
  if (row.error) {
    if (!parts.error) parts.error = appendMessage("error", row.error, "Invocation failed");
    else parts.error.text.textContent = row.error;
  } else if (parts.error) {
    parts.error.wrapper.remove();
    delete parts.error;
  }
  if (parts.assistant && !ACTIVE.has(row.status)) parts.assistant.wrapper.classList.remove("streaming");
}
function renderTranscript() {
  for (const parts of state.messages.values()) cancelTextAnimation(parts);
  state.messages.clear();
  elements.chatLog.replaceChildren(elements.emptyState);
  elements.emptyState.hidden = false;
  for (const row of state.runs) renderRun(row);
  if (state.pending && !state.runs.some(row => row.invocation_id === state.pending.invocation_id)) {
    appendMessage("user", inputText(state.pending.input), "You");
  }
  renderEvents();
}
function latestRun() { return state.runs[state.runs.length - 1]; }
function mergeRun(row) {
  const index = state.runs.findIndex(item => item.id === row.id);
  const previous = state.runs[index];
  if (previous && previous.attempt_id === row.attempt_id && !row.stream && ACTIVE.has(row.status)) {
    row = { ...row, stream: previous.stream, display_text: previous.display_text };
  }
  if (index < 0) state.runs.push(row);
  else state.runs[index] = row;
  const nearBottom = elements.chatLog.scrollHeight - elements.chatLog.scrollTop - elements.chatLog.clientHeight < 100;
  renderRun(row, true);
  if (nearBottom) elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
  $("run-details").textContent = JSON.stringify(row, null, 2);
  renderEvents();
  renderLifecycle();
}
function eventsForRun(row) {
  const events = new Map();
  const add = (id, category, title, payload, timestamp, url) => {
    const key = JSON.stringify([row.id, id]);
    events.set(key, { key, runId: row.id, category, title, payload, timestamp, url });
  };
  add("invocation", "invocation", `Invocation ${row.status}`, {
    run_id: row.id, invocation_id: row.invocation_id, status: row.status, stage: row.stage, ...(row.error ? { error: row.error } : {}),
  }, row.updated_at);
  if (ACTIVE.has(row.status) && row.stage) add("stage", String(row.stage).toLowerCase().includes("running agent") ? "invocation" : "infrastructure", row.stage, { stage: row.stage, run_id: row.id }, row.updated_at);
  const lifecycle = row.result?.lifecycle || row.lifecycle;
  const names = { sandbox_start: "Sandbox start", source_copy: "Artifact / source copy", install_build: "Install / build", agent_run: "Agent run" };
  for (const [key, name] of Object.entries(names)) {
    const step = lifecycle?.steps?.[key];
    if (!step || step.status === "pending") continue;
    const status = step.status === "running" && !ACTIVE.has(row.status) ? "unconfirmed" : step.status;
    const duration = ["completed", "failed"].includes(status) && measured(step.duration_ms) ? ` · ${seconds(step.duration_ms)}` : "";
    add(`step:${key}`, key === "agent_run" ? "invocation" : "infrastructure", `${name}: ${status}${duration}`, step);
  }
  const result = row.result?.result;
  if (!ACTIVE.has(row.status)) {
    if (Array.isArray(result?.tool_events)) {
      for (const event of result.tool_events) {
        if (typeof event?.call_id !== "string" || !event.call_id || typeof event.name !== "string" || !["completed", "failed"].includes(event.status)) continue;
        add(`tool:${event.call_id}`, "tools", `${event.name} · ${event.status}`, event, event.timestamp);
      }
    } else if (Array.isArray(result?.tool_calls)) {
      result.tool_calls.forEach((name, index) => {
        if (typeof name === "string") add(`tool:${index}`, "tools", `${name} · reported`, { name, reported_at: "invocation completion" });
      });
    }
  }
  const trace = result?.trace;
  const url = safeHttpUrl(trace?.url);
  if (url || typeof trace?.trace_id === "string" && trace.trace_id) add("trace", "traces", url ? "Trace available" : "Trace link unavailable", { trace_id: trace?.trace_id }, undefined, url);
  return [...events.values()];
}
function renderEvents() {
  const events = state.runs.flatMap(eventsForRun);
  const visible = events.filter(event => state.eventFilter === "all" || event.category === state.eventFilter);
  const signature = JSON.stringify([state.eventFilter, events.length, visible]);
  if (signature === state.eventSignature) return;
  state.eventSignature = signature;
  const log = $("event-log");
  const opened = new Set([...log.querySelectorAll("details[open]")].map(detail => detail.parentElement.dataset.eventKey));
  const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  log.replaceChildren();
  $("event-count").textContent = `${visible.length} event${visible.length === 1 ? "" : "s"}`;
  if (!visible.length) {
    const empty = node("div", undefined, "event-empty");
    empty.append(node("strong", events.length ? "No matching events" : "No events in this session"), node("span", events.length ? "Choose another event category." : "Send a message to collect runtime evidence."));
    log.append(empty);
  }
  for (const event of visible) {
    const entry = node("article", undefined, "event-entry");
    entry.dataset.eventKey = event.key; entry.dataset.category = event.category; entry.dataset.runId = event.runId;
    if (event.payload.status === "failed") entry.dataset.status = "failed";
    const head = node("div", undefined, "event-entry-header");
    head.append(node("span", { invocation: "Invocation", infrastructure: "Infrastructure", tools: "Tool", traces: "Trace" }[event.category], "event-type"));
    const timestamp = Date.parse(event.timestamp);
    if (Number.isFinite(timestamp)) head.append(node("span", formatClock(new Date(timestamp)), "event-time"));
    const details = node("details", undefined, "event-payload");
    details.open = opened.has(event.key);
    details.append(node("summary", "Event details"), node("pre", JSON.stringify(event.payload, null, 2)));
    entry.append(head, node("p", event.title, "event-title"), details);
    if (event.url) entry.append(externalLink("View trace ↗", event.url, "event-trace"));
    log.append(entry);
  }
  if (nearBottom) log.scrollTop = log.scrollHeight;
}
function seconds(value) { return `${(Math.max(0, value) / 1000).toFixed(1)} s`; }
function liveStatus(row) {
  const stage = String(row?.stage || "").toLowerCase();
  if (stage.includes("upload")) return "Uploading source";
  if (stage.includes("install") || stage.includes("build")) return "Installing dependencies";
  if (stage.includes("creating")) return "Creating sandbox";
  if (stage.includes("resum")) return "Starting sandbox";
  if (stage.includes("running agent")) return "Running";
  return row?.stage || "Queued";
}
const TIMING_STEPS = ["sandbox_start", "source_copy", "install_build", "handler_prepare", "agent_run"];
const TIMING_HINTS = {
  sandbox_start: "Session lookup, create or resume, connection and environment checks.",
  source_copy: "Copy the published source revision into this sandbox.",
  install_build: "Install locked dependencies and build the agent wheel inside the sandbox.",
  handler_prepare: "Start the resident worker, import the handler and run its optional preparation hook.",
  agent_run: "Worker startup, imports, handler work and stream transport.",
};
const TIMING_LABELS = {
  artifact_download: "Artifact download", artifact_verify_extract: "Verify / extract artifact",
  agent_venv: "Agent environment", offline_install: "Offline wheel install",
  extract: "Extract source", toolchain: "Toolchain", toolchain_venv: "Toolchain environment",
  toolchain_download: "Toolchain download", dependency_install: "Dependencies",
  project_build: "Build wheel", project_install: "Install project", import_check: "Import check",
  prepare: "Handler preparation", agent_execution: "Agent execution",
  import_cache: "Import cache setup", framework_import: "Framework imports", project_import: "Agent project imports",
  agentkit_import: "AgentKit imports", tracing_import: "Tracing imports",
  bindings: "Source bindings", configuration: "Agent configuration", session_open: "Session setup",
  agent_build: "Agent construction", mcp_discovery: "MCP tool discovery",
};
function measured(value) { return Number.isFinite(value) && value >= 0; }
function legacySteps(row) {
  const result = row?.result;
  const timing = result?.timings_ms || {};
  const steps = {};
  if (result) {
    const startKeys = ["source_pack", "registry_connect_and_lock", "sandbox_lookup", "sandbox_create_or_reconcile", "sandbox_resume", "remote_connect", "environment_probe"];
    const startDurations = startKeys.map(key => timing[key]).filter(measured);
    steps.sandbox_start = { status: "completed", ...(startDurations.length ? { duration_ms: startDurations.reduce((sum, value) => sum + value, 0) } : {}) };
    for (const [key, metric] of [["source_copy", "source_transfer"], ["install_build", "bootstrap"], ["agent_run", "worker_process_and_execution"]]) {
      steps[key] = result.environment_reused === true && key !== "agent_run"
        ? { status: "reused" }
        : { status: "completed", duration_ms: timing[metric] };
    }
    // The marker's installation metrics are historical on a reused turn.
    if (result.environment_reused === false) steps.install_build.details_ms = result.initial_bootstrap_timings_ms;
    steps.agent_run.details_ms = result.result?.worker_timings_ms;
  } else if (row && ACTIVE.has(row.status)) {
    const stage = String(row.stage || "").toLowerCase();
    const index = stage.includes("running agent") ? 3 : stage.includes("install") || stage.includes("build") ? 2 : stage.includes("upload") ? 1 : row.status === "queued" ? -1 : 0;
    TIMING_STEPS.forEach((key, position) => { steps[key] = { status: position < index ? "completed" : position === index ? "running" : "pending" }; });
  }
  return steps;
}
function renderTimings(row) {
  const preparation = state.preparation;
  const showPreparation = preparation && (preparing() || !row || Date.parse(preparation.created_at) > Date.parse(row.created_at));
  if (showPreparation) row = { ...preparation, status: ["queued", "preparing", "submitting"].includes(preparation.status) ? "running" : preparation.status };
  const lifecycle = row?.result?.lifecycle || row?.lifecycle;
  const steps = lifecycle?.steps || legacySteps(row);
  const active = row && ACTIVE.has(row.status);
  let hasDetails = false;
  for (const key of TIMING_STEPS) {
    const id = key.replaceAll("_", "-");
    const step = steps[key];
    if (key === "handler_prepare") $("step-handler-prepare").hidden = !step;
    let status = step?.status || (active || state.pending ? "pending" : row ? "unavailable" : "empty");
    const abandoned = status === "running" && row && !active;
    if (abandoned) status = row.status === "interrupted" ? "interrupted" : "unavailable";
    if (status === "pending" && row && !active) status = "not_reached";
    let text = { empty: "—", pending: "Pending", reused: "Reused", skipped: "Skipped", completed: "Completed", failed: "Failed", interrupted: "Interrupted", unavailable: "Unavailable", not_reached: "Not reached" }[status] || "Unavailable";
    if (["completed", "failed"].includes(status) && measured(step?.duration_ms)) text = `${status === "failed" ? "Failed · " : ""}${seconds(step.duration_ms)}`;
    if (status === "running") {
      const started = Date.parse(step.started_at);
      text = Number.isFinite(started) ? `Running · ${seconds(Date.now() - started)} elapsed` : "Running";
    }
    $(`${id}-duration`).textContent = text;
    $(`step-${id}`).dataset.status = status;
    $(`${id}-detail`).textContent = abandoned ? "Execution ended before this timing was confirmed."
      : status === "reused" ? key === "source_copy" ? "Artifact or source already present; no copy this turn." : "Installed environment reused; no build this turn."
      : status === "not_reached" ? "This step was not reached."
      : status === "skipped" ? "This step was not needed."
      : TIMING_HINTS[key];
    const details = ["completed", "failed"].includes(status) && step?.details_ms && typeof step.details_ms === "object"
      ? Object.entries(step.details_ms).filter(([, value]) => measured(value)) : [];
    $(`${id}-details`).hidden = !details.length;
    const metrics = $(`${id}-metrics`);
    metrics.replaceChildren();
    for (const [name, value] of details) metrics.append(node("dt", TIMING_LABELS[name] || name.replaceAll("_", " ")), node("dd", seconds(value)));
    hasDetails ||= details.length > 0;
  }
  $("timing-detail").hidden = !hasDetails;
  const startup = lifecycle?.preparation_ms ?? row?.result?.timings_ms?.environment_startup;
  if (measured(startup)) {
    $("startup-duration").textContent = seconds(startup);
    $("startup-detail").textContent = showPreparation ? "Session preparation includes sandbox startup, installation and handler initialization. No invocation was submitted." : "Preparation before agent run. Worker startup, imports and response time follow below.";
  } else if (state.pending || active && steps.agent_run?.status !== "running" && steps.agent_run?.status !== "completed") {
    const started = state.pending?.created_at || Date.parse(row?.created_at);
    $("startup-duration").textContent = Number.isFinite(started) ? `${seconds(Date.now() - started)} elapsed` : "In progress";
    $("startup-detail").textContent = "Elapsed since submission, including queue time; confirmed preparation duration follows.";
  } else {
    $("startup-duration").textContent = active ? "Ready" : row ? "Unavailable" : "—";
    $("startup-detail").textContent = active ? "Agent running; preparation timing is unavailable." : row ? "No confirmed preparation timing for this invocation." : "Launches on the first message.";
  }
  const firstText = row?.stream?.first_token_ms;
  $("first-text-duration").textContent = showPreparation ? "—" : measured(firstText) ? seconds(firstText) : active ? "Waiting" : state.pending ? "Pending" : row ? "Unavailable" : "—";
  $("first-text-detail").hidden = showPreparation || !row && !state.pending;
  const worker = row?.result?.worker;
  const summary = $("worker-summary");
  summary.hidden = !["persistent", "oneshot"].includes(worker?.mode);
  summary.textContent = "";
  summary.title = "";
  if (worker?.mode === "oneshot") summary.textContent = "Fresh process";
  if (worker?.mode === "persistent") {
    const parts = [worker.reused ? "Reused worker" : "Started worker"];
    if (Number.isInteger(worker.pid) && worker.pid > 0) parts.push(`PID ${worker.pid}`);
    if (Number.isInteger(worker.invocation_count) && worker.invocation_count > 0) parts.push(`turn ${worker.invocation_count}`);
    summary.textContent = parts.join(" · ");
    if (typeof worker.id === "string") summary.title = `Worker ${worker.id}`;
  }
}
function renderLifecycle() {
  const pool = state.warmPool;
  $("warm-pool-status").hidden = !state.config?.sandbox_warm_pool_size;
  $("warm-pool-status").textContent = pool
    ? `Warm pool: ${pool.ready} ready · ${pool.preparing} preparing · ${pool.retiring} retiring · maximum ${pool.maximum}${pool.error ? " · replenishment needs attention" : ""}`
    : "Warm pool: checking readiness…";
  const assigned = state.sandbox?.warm_pool;
  $("warm-pool-allocation").hidden = !assigned;
  $("warm-pool-allocation").textContent = "This session received a prepared sandbox from the warm pool.";
  const row = latestRun();
  const active = row && ACTIVE.has(row.status);
  const operation = stopping();
  const labels = {
    not_created: "Not created", creating: "Creating sandbox", bootstrapping: "Installing dependencies",
    ready: "Ready", running: "Running", stopped: "Stopped", failed: "Failed", missing: "Missing",
    unknown: "Unknown", terminated: "Stopped",
  };
  let label = state.loading ? "Loading session…" : state.sandbox ? labels[state.sandbox.status] || "Unknown" : "Checking sandbox…";
  if (row?.result?.sandbox_id && state.sandbox?.status === "not_created") label = "Checking sandbox…";
  if (preparing()) label = state.preparation?.stage || "Preparing session…";
  else if (state.preparation?.error && !row) label = "Preparation failed";
  if (active) label = liveStatus(row);
  if (state.pending) label = state.busy ? "Submitting invocation…" : "Awaiting confirmation";
  if (operation) label = "Stopping…";
  $("sandbox-status").textContent = label;
  $("sandbox-status").className = `run-status${active || operation ? " busy" : state.sandbox?.status === "failed" ? " error" : ""}`;
  $("run-status").textContent = active ? liveStatus(row) : state.pending ? "Awaiting confirmation" : row?.status === "failed" || row?.status === "interrupted" ? "Invocation failed" : state.loading ? "Loading…" : state.agent ? "Ready" : "Choose agent source";
  if (preparing()) $("run-status").textContent = label;
  else if (needsPreparation() && !active) $("run-status").textContent = state.preparation?.error ? "Preparation failed" : "Prepare session to chat";
  $("run-status").className = `run-status${active || initializingSession() ? " busy" : row?.error ? " error" : ""}`;
  $("sandbox-id").textContent = state.sandbox?.sandbox_id || row?.result?.sandbox_id || "—";
  $("principal").textContent = state.sandbox?.execution_principal || state.config?.service_principal || "—";
  renderTimings(state.pending && state.pending.invocation_id !== row?.invocation_id ? null : row);
  const stop = state.context && savedObject(stopKey(state.context));
  $("stop-duration").textContent = operation ? `${seconds(Date.now() - operation.started)} elapsed` : Number.isFinite(stop?.timings_ms?.total) ? seconds(stop.timings_ms.total) : stop?.unconfirmed ? `${seconds(stop.elapsed_ms)} · unconfirmed` : "—";
  $("stop-detail").hidden = !stop && !operation;
  $("stop-detail").textContent = operation ? "Waiting for shutdown confirmation." : stop?.unconfirmed ? "Shutdown was not confirmed. Reconnect to check status." : stop ? "Last confirmed sandbox shutdown." : "";
  updateControls();
}
function closeWatch() {
  state.source?.close();
  state.source = null;
  clearTimeout(state.preparationTimer);
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
}
async function refreshSandbox(ctx) {
  if (!current(ctx) || stopping(ctx)) return;
  const key = stopKey(ctx);
  const request = nextSandboxRequest(ctx);
  const fresh = () => current(ctx) && state.sandboxRequests.get(key) === request && !stopping(ctx);
  try {
    const data = await api(`/api/agents/${encodeURIComponent(ctx.agentId)}/sandbox?${new URLSearchParams({ session_id: ctx.sessionId })}`, undefined, ctx, API_TIMEOUTS.status);
    if (!fresh()) return;
    state.sandbox = data;
    if (data.preparation) state.preparation = data.preparation;
    state.sandboxes.set(key, data);
  } catch (error) {
    if (!fresh()) return;
    state.sandbox = { ...state.sandboxes.get(key), status: "unknown" };
    state.sandboxes.set(key, state.sandbox);
    notice(`Sandbox status unavailable: ${error.message}`, "reload");
  }
  if (current(ctx)) renderLifecycle();
}
function renderSessions() {
  const target = $("session-list");
  target.replaceChildren();
  const query = $("session-search-input").value.trim().toLowerCase();
  const sessions = [...state.sessions];
  if (state.context && !sessions.some(row => row.session_id === state.context.sessionId)) sessions.unshift({ session_id: state.context.sessionId, status: "New chat" });
  for (const session of sessions.filter(row => row.session_id.toLowerCase().includes(query))) {
    const selected = session.session_id === state.context?.sessionId;
    const button = node("button", undefined, `session-item session-open-button${selected ? " current" : ""}`);
    button.type = "button";
    if (selected) button.setAttribute("aria-current", "true");
    const top = node("div", undefined, "session-item-top");
    top.append(node("span", selected ? "Current session" : "Session", "session-item-title"), node("span", session.status, "session-item-time"));
    button.append(top, node("span", session.session_id, "session-item-id"));
    button.addEventListener("click", () => { if (!selected) void selectSession(session.session_id); });
    target.append(button);
  }
  if (!target.children.length) target.append(node("div", query ? "No sessions match your search." : "No sessions yet.", "state-empty"));
}
async function refreshSessions(ctx) {
  const rows = await api(`/api/agents/${encodeURIComponent(ctx.agentId)}/sessions`, undefined, ctx);
  if (!current(ctx)) return;
  state.sessions = rows;
  renderSessions();
}
async function history(ctx) { return api(contextUrl(ctx, "/api/runs"), undefined, ctx); }
async function beginPreparation(ctx) {
  if (!current(ctx) || !supportsPreparation() || preparing()) return;
  nextSandboxRequest(ctx); // A pre-resume status read cannot overwrite new preparation state.
  state.preparation = { status: "submitting", stage: "Requesting session preparation" };
  notice(); renderLifecycle(); selectEvidenceTab("sandbox");
  try {
    const result = await api(`/api/agents/${encodeURIComponent(ctx.agentId)}/sessions`, { session_id: ctx.sessionId, retry: true }, ctx, 20000);
    if (!current(ctx)) return;
    state.preparation = result;
    await pollPreparation(ctx);
    void refreshSessions(ctx).catch(() => {});
  } catch (error) {
    if (!current(ctx)) return;
    state.preparation = { status: "unknown", stage: "Preparation response not confirmed" };
    notice(`Preparation response unavailable: ${error.message}. Reconnect to check this same session.`, "prepare-status");
    renderLifecycle();
  }
}
async function pollPreparation(ctx) {
  if (!current(ctx)) return;
  clearTimeout(state.preparationTimer);
  try {
    const result = await api(`/api/agents/${encodeURIComponent(ctx.agentId)}/sessions/${encodeURIComponent(ctx.sessionId)}`, undefined, ctx, API_TIMEOUTS.status);
    if (!current(ctx)) return;
    state.preparation = result;
    if (result.sandbox_id) state.sandbox = { ...state.sandbox, sandbox_id: result.sandbox_id };
    if (["queued", "preparing"].includes(result.status)) {
      state.preparationTimer = setTimeout(() => { void pollPreparation(ctx); }, 1000);
    } else {
      await refreshSandbox(ctx);
      if (!current(ctx)) return;
      if (result.error) notice(result.error);
      else notice();
    }
  } catch (error) {
    if (!current(ctx)) return;
    state.preparation = { ...state.preparation, status: "unknown" };
    notice(`Preparation status unavailable: ${error.message}`, "prepare-status");
  }
  if (current(ctx)) renderLifecycle();
}
async function newSession() {
  if (!state.agent) return;
  const id = crypto.randomUUID();
  await selectSession(id);
  if (state.context?.sessionId === id) await beginPreparation(state.context);
}
async function selectSession(sessionId) {
  state.context?.controller.abort();
  closeWatch();
  const ctx = { agentId: state.agent.id, sessionId, controller: new AbortController() };
  state.context = ctx;
  state.preparation = null;
  state.runs = []; state.sessions = []; state.sandbox = state.sandboxes.get(stopKey(ctx)) || null;
  state.pending = savedObject(pendingKey(ctx));
  state.loading = true; state.historyUnavailable = false;
  setBusy(false); notice();
  remember(`sandbox-chat:session:${ctx.agentId}`, sessionId);
  $("session-id").textContent = sessionId;
  $("prompt-input").value = "";
  $("run-details").textContent = "No invocation yet.";
  renderTranscript(); renderSessions(); renderLifecycle();
  // Reads never create a sandbox. History and session metadata load independently.
  const results = await Promise.allSettled([history(ctx), refreshSessions(ctx), refreshSandbox(ctx)]);
  if (!current(ctx)) return;
  state.loading = false;
  if (results[0].status === "rejected") {
    // Keep the composer blocked until history is known: an invocation may still be active.
    state.historyUnavailable = true;
    notice(`Could not load conversation: ${results[0].reason.message}`, "reload");
    renderLifecycle();
    return;
  }
  state.runs = results[0].value;
  const confirmed = state.pending && state.runs.find(row => row.invocation_id === state.pending.invocation_id);
  if (confirmed) clearPending(ctx);
  renderTranscript();
  const row = latestRun();
  if (row) $("run-details").textContent = JSON.stringify(row, null, 2);
  if (results[1].status === "rejected") notice(`Session list unavailable: ${results[1].reason.message}`, "reload");
  if (state.pending) notice("The last submission was not confirmed. Reconnect with its original invocation ID before sending another message.", "submit");
  else if (row && ACTIVE.has(row.status)) watchRun(row, ctx);
  else if (row?.error) notice(row.error, ["failed", "interrupted"].includes(row.status) ? "failed" : null);
  if (preparing()) void pollPreparation(ctx);
  else if (state.preparation?.error && !row) notice(state.preparation.error);
  renderLifecycle();
}
function clearPending(ctx) {
  try { localStorage.removeItem(pendingKey(ctx)); } catch { /* Keep in-memory identity for this tab. */ }
  if (current(ctx)) state.pending = null;
}
function persistPending(ctx, pending) {
  // Persist before POST so a lost response or reload can reuse the invocation ID.
  localStorage.setItem(pendingKey(ctx), JSON.stringify(pending));
  state.pending = pending;
}
async function terminal(row, ctx) {
  if (!current(ctx)) return;
  closeWatch();
  mergeRun(row); setBusy(false);
  if (row.error) notice(`${row.error}${["failed", "interrupted"].includes(row.status) ? " Retry explicitly only after checking any partial effects." : ""}`, ["failed", "interrupted"].includes(row.status) ? "failed" : null);
  else notice();
  const results = await Promise.allSettled([refreshSessions(ctx), refreshSandbox(ctx)]);
  if (current(ctx) && results[0].status === "rejected") notice(`Session list unavailable: ${results[0].reason.message}`, "reload");
}
function watchRun(row, ctx, reconnects = 0) {
  if (!current(ctx)) return;
  closeWatch(); setBusy(true); mergeRun(row);
  if (!ACTIVE.has(row.status)) { void terminal(row, ctx); return; }
  const source = new EventSource(`/api/runs/${encodeURIComponent(row.id)}/events`);
  state.source = source;
  source.addEventListener("run", event => {
    if (!current(ctx) || state.source !== source) return;
    try {
      const snapshot = JSON.parse(event.data);
      reconnects = 0;
      if (snapshot.id !== row.id || snapshot.agent_id !== ctx.agentId || snapshot.session_id !== ctx.sessionId) throw new Error("Unexpected run snapshot");
      mergeRun(snapshot);
      if (!ACTIVE.has(snapshot.status)) void terminal(snapshot, ctx);
    } catch {
      if (state.source === source) { closeWatch(); void pollRun(row.id, ctx); }
    }
  });
  source.onerror = () => {
    // Check completion immediately, then reconnect to the same run. Replaying
    // its file never submits another invocation. Bound repeated transport errors.
    if (current(ctx) && state.source === source) {
      closeWatch(); void pollRun(row.id, ctx, 0, reconnects + 1);
    }
  };
}
async function pollRun(runId, ctx, failures = 0, reconnects = 0) {
  if (!current(ctx)) return;
  try {
    const row = await api(`/api/runs/${encodeURIComponent(runId)}`, undefined, ctx);
    if (!current(ctx)) return;
    mergeRun(row);
    if (!ACTIVE.has(row.status)) { await terminal(row, ctx); return; }
    state.pollTimer = setTimeout(() => {
      if (reconnects > 0 && reconnects <= 3) watchRun(row, ctx, reconnects);
      else void pollRun(runId, ctx);
    }, 1000);
  } catch (error) {
    if (!current(ctx)) return;
    if (failures < 2) state.pollTimer = setTimeout(() => { void pollRun(runId, ctx, failures + 1, reconnects); }, 1500 * (failures + 1));
    else notice(`Connection lost. The invocation may still be running: ${error.message}`, "monitor");
  }
}
async function submitPending(ctx) {
  const pending = state.pending;
  if (!pending || !current(ctx) || stopping(ctx)) return;
  notice(); setBusy(true); renderLifecycle();
  try {
    const bound = state.config.default_environment;
    const path = bound ? `/api/environments/${encodeURIComponent(bound)}/runs` : "/api/runs";
    // Transport reconnects use retry=false. A failed-run retry creates a new invocation ID.
    const retryFailed = Boolean(pending.retry);
    if (retryFailed) persistPending(ctx, { ...pending, retry: false });
    const row = await api(path, {
      ...(bound ? {} : { agent_id: ctx.agentId }), session_id: ctx.sessionId,
      invocation_id: pending.invocation_id, input: pending.input, retry: retryFailed,
    });
    clearPending(ctx);
    if (!current(ctx)) return;
    $("prompt-input").value = ""; resizeComposer();
    // Replace the optimistic message with the authoritative row, without a second user bubble.
    const index = state.runs.findIndex(item => item.id === row.id);
    if (index < 0) state.runs.push(row); else state.runs[index] = row;
    renderTranscript(); renderSessions();
    watchRun(row, ctx);
  } catch (error) {
    if (!current(ctx)) return;
    setBusy(false);
    notice(`Submission not confirmed: ${error.message}. Reconnect to the same invocation; do not send it as a new message.`, "submit");
    renderLifecycle();
  }
}
async function sendMessage(event) {
  event.preventDefault();
  const text = $("prompt-input").value.trim();
  const ctx = state.context;
  if (!text || !ctx || needsPreparation() || state.busy || state.loading || state.historyUnavailable || state.pending || stopping(ctx)) return;
  try {
    const input = { message: text };
    if (state.model) input.model = state.model;
    persistPending(ctx, { invocation_id: crypto.randomUUID(), input, created_at: Date.now(), retry: false });
  } catch {
    notice("Browser storage is unavailable. Enable storage for this page to preserve invocation identity across reconnects.");
    return;
  }
  renderTranscript();
  await submitPending(ctx);
}
async function recover() {
  const ctx = state.context;
  if (!ctx) { void initialize(); return; }
  if (stopping(ctx)) return;
  const action = state.recovery;
  if (action === "prepare-status") { notice(); await pollPreparation(ctx); return; }
  if (action === "submit") { await submitPending(ctx); return; }
  if (action === "monitor") {
    notice(); setBusy(true);
    await pollRun(latestRun().id, ctx);
    return;
  }
  if (action === "failed") {
    const row = latestRun();
    if (!row || !["failed", "interrupted"].includes(row.status)) return;
    try {
      persistPending(ctx, { invocation_id: crypto.randomUUID(), input: row.input, created_at: Date.now(), retry: false });
      await submitPending(ctx);
    } catch (error) { notice(error.message, "failed"); }
    return;
  }
  await selectSession(ctx.sessionId);
}
async function stopSession() {
  const ctx = state.context;
  if (!ctx || stopping(ctx) || $("stop-session").disabled) return;
  const key = stopKey(ctx);
  const operation = { started: Date.now() };
  state.stopOperations.set(key, operation);
  nextSandboxRequest(ctx);
  notice(); renderLifecycle();
  try {
    const result = await api("/sessions/stop", { agent_id: ctx.agentId, session_id: ctx.sessionId }, undefined, API_TIMEOUTS.stop);
    if (!result?.sandbox_id || !["stopped", "terminated"].includes(result.status)) throw new Error("The server did not confirm shutdown.");
    remember(key, JSON.stringify(result));
    nextSandboxRequest(ctx);
    const sandbox = { ...state.sandboxes.get(key), ...result };
    state.sandboxes.set(key, sandbox);
    if (selected(ctx)) { state.sandbox = sandbox; notice(); }
  } catch (error) {
    remember(key, JSON.stringify({ unconfirmed: true, elapsed_ms: Date.now() - operation.started }));
    if (selected(ctx)) notice(`Stop was not confirmed: ${error.message} Reconnect to check status.`, "reload");
  } finally {
    state.stopOperations.delete(key);
    if (selected(ctx)) renderLifecycle();
  }
}
function workingFolderUrl(host, path) {
  try {
    const url = new URL(host);
    if (!["https:", "http:"].includes(url.protocol) || url.username || url.password || !path?.startsWith("/") || path.startsWith("//")) return null;
    url.pathname = "/"; url.search = "";
    const workspacePath = path.startsWith("/Workspace/") ? path.slice("/Workspace".length) : path;
    url.hash = `workspace${workspacePath.split("/").map(encodeURIComponent).join("/")}`;
    return url.href;
  } catch { return null; }
}
function renderAgent() {
  const agent = state.agent;
  $("agent-title").textContent = agent?.name || "Your agent";
  $("brand-sub").textContent = state.config.default_environment || agent?.name || "Sandbox runtime";
  $("active-source").textContent = agent?.source_path || (agent?.source_kind === "bundled" ? "Bundled LangGraph example" : "Choose an agent source to begin.");
  $("source-hash").textContent = agent?.source_hash || "—";
  const source = $("view-session-source");
  source.hidden = !agent;
  if (agent) source.href = `/api/agents/${encodeURIComponent(agent.id)}/source`;
  else source.removeAttribute("href");
  const working = $("open-source-folder");
  const url = agent?.source_kind === "workspace" && workingFolderUrl(state.config.workspace, agent.source_path);
  working.hidden = !url;
  if (url) working.href = url; else working.removeAttribute("href");
  const git = agent?.git_source;
  $("active-git").hidden = !git;
  $("active-git").textContent = git ? `Git ${git.commit} · ${git.requested_revision}${git.subdirectory ? ` · ${git.subdirectory}` : ""}` : "";
  $("agent-select").value = agent?.id || "";
}
async function selectAgent(agent) {
  if (state.agent?.id !== agent.id) state.model = "";
  state.agent = agent;
  renderModels();
  remember("sandbox-chat:agent", agent.id);
  renderAgent();
  await selectSession(read(`sandbox-chat:session:${agent.id}`) || crypto.randomUUID());
}
async function refreshAgents() {
  const agents = await api("/api/agents");
  state.agents = agents;
  const select = $("agent-select");
  select.replaceChildren(node("option", "Select a revision"));
  select.firstElementChild.value = "";
  for (const agent of agents) {
    const option = node("option", `${agent.name} · ${agent.source_hash.slice(0, 10)}`);
    option.value = agent.id; select.append(option);
  }
  select.value = state.agent?.id || "";
}
async function loadSource(event) {
  event.preventDefault();
  $("load-source").disabled = true;
  $("source-notice").textContent = "Saving source revision…";
  try {
    const agent = await api("/api/agents", {
      name: $("agent-name").value.trim(), source_kind: $("source-kind").value,
      source_path: $("source-kind").value === "bundled" ? "" : $("source-path").value.trim(), entrypoint: $("entrypoint").value.trim(),
    });
    $("source-notice").textContent = "Source saved. Send a message to launch a sandbox.";
    await refreshAgents(); await selectAgent(agent);
  } catch (error) { $("source-notice").textContent = error.message; }
  finally { $("load-source").disabled = false; }
}
function resizeComposer() {
  const input = $("prompt-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  updateControls();
}
async function initialize() {
  try {
    state.config = await api("/api/config");
    void refreshWarmPool();
    void loadModels();
    renderCapabilities();
    const shared = state.config.sandbox_scope === "environment";
    $("scope-description").textContent = shared ? "Legacy environment: sessions share one sandbox, with separate conversations." : "New Chat prepares a dedicated sandbox and resident agent before you send a message.";
    $("empty-description").textContent = shared ? "Chat with the installed agent in this environment's shared sandbox." : "Choose Prepare session or New Chat to start a sandbox, install the source and initialize the agent. You can draft a message while it prepares.";
    $("stop-session").textContent = shared ? "Stop shared sandbox" : "Stop this session";
    $("stop-scope").hidden = !shared;
    $("stop-scope").textContent = "Stopping affects every session in this legacy environment.";
    $("source-selection").hidden = Boolean(state.config.default_environment);
    $("source-path").value = state.config.example_workspace_path || "";
    $("entrypoint").value = state.config.default_entrypoint || "sandbox_agent.worker:run";
    $("viewer-value").textContent = "Sandbox runtime";
    await refreshAgents();
    const defaultId = state.config.default_agent_id;
    const selected = defaultId ? state.agents.find(agent => agent.id === defaultId) : state.agents.find(agent => agent.id === read("sandbox-chat:agent")) || state.agents[0];
    if (defaultId && !selected) throw new Error("The configured agent revision is unavailable.");
    if (selected) await selectAgent(selected);
    else {
      $("source-selection").open = true;
      renderAgent(); renderLifecycle();
    }
  } catch (error) {
    $("brand-sub").textContent = "Connection unavailable";
    notice(error.message, "reload");
  }
}
$("composer").addEventListener("submit", event => { void sendMessage(event); });
$("model-select").addEventListener("change", event => { state.model = event.target.value; });
$("reload-models").addEventListener("click", () => { void loadModels(); });
for (const name of ["events", "sandbox"]) {
  $(`${name}-tab`).addEventListener("click", () => selectEvidenceTab(name));
  $(`${name}-tab`).addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    selectEvidenceTab(event.key === "Home" ? "events" : event.key === "End" ? "sandbox" : name === "events" ? "sandbox" : "events", true);
  });
}
document.querySelectorAll(".capability").forEach(button => button.addEventListener("click", () => {
  const expanded = button.getAttribute("aria-expanded") !== "true";
  button.setAttribute("aria-expanded", String(expanded));
  button.closest(".capability-item").classList.toggle("expanded", expanded);
  $(button.getAttribute("aria-controls")).hidden = !expanded;
}));
document.querySelectorAll(".event-filter").forEach(button => button.addEventListener("click", () => {
  state.eventFilter = button.dataset.filter;
  document.querySelectorAll(".event-filter").forEach(filter => {
    const selected = filter === button;
    filter.classList.toggle("active", selected);
    filter.setAttribute("aria-pressed", String(selected));
  });
  renderEvents();
}));
$("prompt-input").addEventListener("input", resizeComposer);
$("prompt-input").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!$("send-button").disabled) $("composer").requestSubmit();
  }
});
$("new-session").addEventListener("click", () => { void newSession(); });
$("prepare-session").addEventListener("click", () => { if (state.context) void beginPreparation(state.context); });
$("refresh-session").addEventListener("click", () => { if (state.context) void selectSession(state.context.sessionId); });
$("session-search-input").addEventListener("input", renderSessions);
$("retry-run").addEventListener("click", () => { void recover(); });
$("stop-session").addEventListener("click", () => { void stopSession(); });
$("agent-select").addEventListener("change", event => {
  const agent = state.agents.find(item => item.id === event.target.value);
  if (agent) void selectAgent(agent);
});
$("refresh-agents").addEventListener("click", () => { void refreshAgents().catch(error => { $("source-notice").textContent = error.message; }); });
$("source-form").addEventListener("submit", event => { void loadSource(event); });
$("source-kind").addEventListener("change", () => {
  const bundled = $("source-kind").value === "bundled";
  $("path-field").hidden = bundled; $("source-path").required = !bundled;
});
document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => {
  $("prompt-input").value = button.dataset.prompt; resizeComposer(); $("prompt-input").focus();
}));
async function refreshWarmPool() {
  if (!state.config?.sandbox_warm_pool_size || document.hidden) return;
  try { state.warmPool = await api("/api/warm-pool", undefined, undefined, API_TIMEOUTS.status); }
  catch { state.warmPool = null; }
  if (state.context) renderLifecycle();
}
setInterval(() => { void refreshWarmPool(); }, 10000);
setInterval(() => { if (state.context) renderLifecycle(); }, 500);
window.addEventListener("pagehide", closeWatch);
void initialize();
