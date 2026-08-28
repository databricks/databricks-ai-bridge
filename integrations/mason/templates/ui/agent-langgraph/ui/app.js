const elements = {
  approvalPanel: document.querySelector("#approval-panel"),
  approvalSummary: document.querySelector("#approval-summary"),
  approveAction: document.querySelector("#approve-action"),
  backgroundMode: document.querySelector("#background-mode-value"),
  backgroundStatus: document.querySelector("#background-status"),
  chatLog: document.querySelector("#chat-log"),
  clearEvents: document.querySelector("#clear-events"),
  composer: document.querySelector("#composer"),
  connectionStatus: document.querySelector("#connection-status"),
  copySession: document.querySelector("#copy-session"),
  durabilityAttempt: document.querySelector("#durability-attempt"),
  durabilityExecution: document.querySelector("#durability-execution"),
  durabilityHeartbeat: document.querySelector("#durability-heartbeat"),
  durabilityMode: document.querySelector("#durability-mode-value"),
  durabilityOwner: document.querySelector("#durability-owner"),
  durabilityStatus: document.querySelector("#durability-status"),
  emptyState: document.querySelector("#empty-state"),
  environmentBadge: document.querySelector("#environment-badge"),
  eventLog: document.querySelector("#event-log"),
  identitySummary: document.querySelector("#identity-summary"),
  identityValue: document.querySelector("#identity-value"),
  heartbeatMode: document.querySelector("#heartbeat-mode-value"),
  heartbeatStatus: document.querySelector("#heartbeat-status"),
  memoryFact: document.querySelector("#memory-fact"),
  memoryHelp: document.querySelector("#memory-help"),
  memoryMode: document.querySelector("#memory-mode-value"),
  memoryPath: document.querySelector("#memory-path"),
  memoryQuery: document.querySelector("#memory-query"),
  memoryResults: document.querySelector("#memory-results"),
  memoryStatus: document.querySelector("#memory-status"),
  promptInput: document.querySelector("#prompt-input"),
  askMemory: document.querySelector("#ask-memory"),
  searchMemory: document.querySelector("#search-memory"),
  recoverySteps: document.querySelector("#recovery-steps"),
  recoveryStatus: document.querySelector("#recovery-status"),
  refreshConfig: document.querySelector("#refresh-config"),
  refreshSession: document.querySelector("#refresh-session"),
  rejectAction: document.querySelector("#reject-action"),
  rejectSession: document.querySelector("#reject-session"),
  rememberButton: document.querySelector("#remember-button"),
  resumeSession: document.querySelector("#resume-session"),
  runStatus: document.querySelector("#run-status"),
  sendButton: document.querySelector("#send-button"),
  sessionId: document.querySelector("#session-id"),
  sessionItems: document.querySelector("#session-items"),
  sessionMode: document.querySelector("#session-mode-value"),
  sessionStatus: document.querySelector("#session-status"),
  sessionStoreLabel: document.querySelector("#session-store-label"),
  startApp: document.querySelector("#start-app"),
  stopApp: document.querySelector("#stop-app"),
  streamingMode: document.querySelector("#streaming-mode-value"),
  streamingStatus: document.querySelector("#streaming-status"),
  viewerValue: document.querySelector("#viewer-value"),
};

const state = {
  busy: false,
  config: null,
  draft: null,
  draftText: "",
  events: [],
  instanceId: null,
  lastDurabilityAttempt: 0,
  lastDurabilityHeartbeat: "",
  lastRecoveryStatus: "",
  lastAssistantText: "",
  managedSessionId: "",
  mode: "streaming",
  pendingInterrupt: null,
  recovery: null,
  recoveryPollTimer: null,
  recoverySessionId: "",
  recoverySignature: "",
  sessionId: "",
};

function ensureSessionId() {
  if (!state.sessionId) throw new Error("The routing session is not initialized yet.");
  return state.sessionId;
}

function setSessionId(value) {
  const nextSessionId = String(value || "");
  if (!nextSessionId) return;
  if (state.sessionId !== nextSessionId) state.managedSessionId = "";
  state.sessionId = nextSessionId;
  elements.sessionId.textContent = state.sessionId;
}

function setConnection(status, label) {
  elements.connectionStatus.dataset.state = status;
  elements.connectionStatus.querySelector("span:last-child").textContent = label;
}

function setStatus(label, type = "ready") {
  elements.runStatus.textContent = label;
  elements.runStatus.className = `run-status ${type === "ready" ? "" : type}`.trim();
}

function setBusy(busy, label = "Working") {
  state.busy = busy;
  elements.chatLog.setAttribute("aria-busy", String(busy));
  elements.sendButton.disabled = busy;
  elements.promptInput.disabled = busy;
  elements.approveAction.disabled = busy;
  elements.rejectAction.disabled = busy;
  elements.rememberButton.disabled = busy || !state.config?.memory.enabled;
  elements.searchMemory.disabled = busy || !state.config?.memory.enabled;
  elements.askMemory.disabled = busy || !state.config?.memory.enabled;
  elements.refreshSession.disabled = busy || !state.config?.session.history;
  elements.resumeSession.disabled = busy || !state.config?.session.durable;
  elements.rejectSession.disabled = busy || !state.config?.session.durable;
  updateRecoveryControls();
  if (busy) setStatus(label, "busy");
  else if (!elements.runStatus.classList.contains("error")) setStatus("Ready");
}

function setCapability(element, enabled) {
  element.classList.toggle("enabled", Boolean(enabled));
  element.classList.toggle("disabled", !enabled);
}

function recoveryInProgress() {
  return Boolean(
    state.recovery?.worker_active
      || state.recovery?.owner_active
      || state.recovery?.needs_resume,
  );
}

function updateRecoveryControls() {
  const canStart = state.config?.recovery.enabled
    && (!state.recovery || state.recovery.status === "not_started" || state.recovery.needs_resume);
  const checkpointedSteps = state.recovery?.outputs?.length || 0;
  const canStop = state.config?.app_control.stop_enabled
    && (state.pendingInterrupt || (state.recovery?.worker_active && checkpointedSteps >= 2));
  elements.startApp.disabled = state.busy || !canStart;
  elements.stopApp.disabled = state.busy || !canStop;
}

function formatJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function addEvent(type, payload) {
  state.events.unshift({ type, payload, at: new Date() });
  state.events = state.events.slice(0, 60);
  elements.eventLog.replaceChildren();
  for (const event of state.events) {
    const entry = document.createElement("div");
    entry.className = "event-entry";
    const header = document.createElement("div");
    header.className = "event-entry-header";
    const name = document.createElement("span");
    name.textContent = event.type;
    const time = document.createElement("span");
    time.textContent = event.at.toLocaleTimeString();
    const body = document.createElement("pre");
    body.textContent = formatJson(event.payload);
    header.append(name, time);
    entry.append(header, body);
    elements.eventLog.append(entry);
  }
}

function normalizeRole(message) {
  const value = String(message?.role || message?.type || "assistant").toLowerCase();
  if (["human", "user"].includes(value)) return "user";
  if (["tool", "function"].includes(value)) return "tool";
  if (["system", "developer"].includes(value)) return "system";
  return "assistant";
}

function extractText(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (typeof part?.text === "string") return part.text;
        if (typeof part?.content === "string") return part.content;
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return typeof content === "object" ? formatJson(content) : String(content);
}

function hideEmptyState() {
  elements.emptyState.hidden = true;
}

function appendMessage(role, content, label) {
  hideEmptyState();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "YOU" : role === "assistant" ? "AI" : role === "tool" ? "TOOL" : "!";
  const body = document.createElement("div");
  body.className = "message-body";
  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = label || (role === "user" ? "You" : role === "assistant" ? "Agent" : role);
  const text = document.createElement("div");
  text.className = "message-content";
  text.textContent = content;
  body.append(meta, text);
  wrapper.append(avatar, body);
  elements.chatLog.append(wrapper);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
  return { wrapper, text };
}

function appendError(error) {
  const message = error instanceof Error ? error.message : String(error);
  appendMessage("error", message, "Request failed");
  setStatus("Error", "error");
  addEvent("error", { message });
}

function startDraft() {
  if (state.draft) return state.draft;
  const draft = appendMessage("assistant", "", "Agent · streaming");
  draft.wrapper.classList.add("streaming");
  state.draft = draft;
  state.draftText = "";
  return draft;
}

function appendDelta(content) {
  const text = extractText(content);
  if (!text) return;
  const draft = startDraft();
  state.draftText += text;
  state.lastAssistantText = state.draftText;
  draft.text.textContent = state.draftText;
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

function finishDraft(finalText = "") {
  if (!state.draft) return false;
  if (finalText) {
    state.draftText = finalText;
    state.lastAssistantText = finalText;
    state.draft.text.textContent = finalText;
  }
  state.draft.wrapper.classList.remove("streaming");
  state.draft.wrapper.querySelector(".message-meta").textContent = "Agent";
  state.draft = null;
  return true;
}

function toolSummary(message) {
  if (message?.name) return `${message.name}\n${extractText(message.content)}`.trim();
  if (Array.isArray(message?.tool_calls) && message.tool_calls.length) {
    return message.tool_calls.map((call) => `${call.name || "tool"}(${formatJson(call.args || {})})`).join("\n");
  }
  return extractText(message?.content) || formatJson(message);
}

function handleAgentMessage(message) {
  const role = normalizeRole(message);
  if (role === "user") return;
  if (role === "assistant") {
    const text = extractText(message?.content);
    if (finishDraft(text)) return;
    if (text) {
      state.lastAssistantText = text;
      appendMessage("assistant", text, "Agent");
    }
    if (message?.tool_calls?.length) appendMessage("tool", toolSummary(message), "Tool request");
    return;
  }
  appendMessage(role, toolSummary(message), role === "tool" ? "Tool result" : "System");
}

function interruptSummary(interrupt) {
  const requests = interrupt?.value?.action_requests || [];
  if (!requests.length) return "The agent paused and needs a decision.";
  return requests
    .map((request) => `${request.name || "tool"} ${formatJson(request.args || {})}`)
    .join(" · ");
}

function handleInterrupt(interrupt) {
  finishDraft();
  state.pendingInterrupt = interrupt;
  elements.approvalSummary.textContent = interruptSummary(interrupt);
  elements.approvalPanel.hidden = false;
  appendMessage("system", interruptSummary(interrupt), "Approval required");
  if (state.config?.app_control.stop_enabled && state.config?.session.durable) {
    elements.recoveryStatus.textContent = "Paused run detected. Stop App, then approve it after restart.";
  }
}

function handleEvent(event) {
  addEvent(event?.type || "event", event);
  if (event?.type === "delta") appendDelta(event.content);
  if (event?.type === "message") handleAgentMessage(event.message);
  if (event?.type === "interrupt") handleInterrupt(event);
  if (event?.error) throw new Error(event.error);
}

function handleOutput(output) {
  for (const item of output || []) {
    if (item?.type === "interrupt") handleInterrupt(item);
    else handleAgentMessage(item);
  }
}

function invocationHeaders() {
  return {
    "Content-Type": "application/json",
  };
}

function invocationPayload(payload) {
  return { ...payload };
}

async function jsonResponse(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.error || `Request failed with ${response.status}`);
  return body;
}

function stateMessage(container, message, kind = "empty") {
  container.replaceChildren();
  const item = document.createElement("div");
  item.className = `state-${kind}`;
  item.textContent = message;
  container.append(item);
}

function renderStateItems(container, items, emptyMessage, renderItem) {
  container.replaceChildren();
  if (!items.length) {
    stateMessage(container, emptyMessage);
    return;
  }
  for (const value of items) {
    const item = document.createElement("article");
    item.className = "state-item";
    const rendered = renderItem(value);
    const title = document.createElement("strong");
    title.textContent = rendered.title;
    const content = document.createElement("p");
    content.textContent = rendered.content || "No content returned.";
    const meta = document.createElement("small");
    meta.textContent = rendered.meta || "";
    item.append(title, content, meta);
    container.append(item);
  }
}

function memoryEntries(payload) {
  return payload?.managed_memory_entries || [];
}

function sessionItems(payload) {
  return payload?.session_items || [];
}

function renderMemoryEntries(entries, emptyMessage = "No matching memory entries.") {
  renderStateItems(elements.memoryResults, entries, emptyMessage, (entry) => ({
    title: entry.path || entry.name || "Memory entry",
    content: extractText(entry.content) || entry.description || "Content is omitted from list responses.",
    meta: [entry.actor_id, entry.session_id, entry.update_time].filter(Boolean).join(" · "),
  }));
}

function renderSessionItems(items) {
  renderStateItems(elements.sessionItems, items, "No transcript items yet.", (item) => {
    const data = item?.data || {};
    return {
      title: String(data.role || data.type || "item"),
      content: extractText(data.content ?? data),
      meta: [item.item_id, item.create_time, data.transport].filter(Boolean).join(" · "),
    };
  });
}

function renderSessionTranscript(items) {
  state.draft = null;
  state.draftText = "";
  state.lastAssistantText = "";
  elements.chatLog.replaceChildren(elements.emptyState);
  elements.emptyState.hidden = false;

  for (const item of items) {
    const data = item?.data || {};
    const storedRole = String(data.role || data.type || "").toLowerCase();
    const content = extractText(data.content ?? data);
    if (!content) continue;
    if (storedRole === "human_decision") {
      appendMessage("system", content, "Human decision");
      continue;
    }
    const role = normalizeRole(data);
    if (role === "assistant") state.lastAssistantText = content;
    appendMessage(
      role,
      role === "tool" ? toolSummary(data) : content,
      role === "user" ? "You" : role === "assistant" ? "Agent" : role === "tool" ? "Tool result" : "System",
    );
  }
}

async function ensureManagedSession() {
  if (!state.config?.session.managed) return null;
  const sessionId = ensureSessionId();
  if (state.managedSessionId === sessionId) return sessionId;
  stateMessage(elements.sessionItems, "Connecting managed session…", "loading");
  const response = await fetch("/api/demo/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const result = await jsonResponse(response);
  state.managedSessionId = sessionId;
  addEvent("session.managed", result);
  return sessionId;
}

async function refreshSession({ hydrateChat = false } = {}) {
  if (!state.config?.session.history) {
    stateMessage(elements.sessionItems, "Session history is not available.");
    return;
  }
  try {
    const sessionId = state.config.session.managed ? await ensureManagedSession() : ensureSessionId();
    const response = await fetch("/api/demo/session/items", { cache: "no-store" });
    const result = await jsonResponse(response);
    const items = sessionItems(result);
    renderSessionItems(items);
    if (hydrateChat) {
      renderSessionTranscript(items);
      for (const interrupt of result.interrupts || []) handleInterrupt(interrupt);
    }
    addEvent("session.items.list", result);
  } catch (error) {
    stateMessage(elements.sessionItems, error instanceof Error ? error.message : String(error), "error");
    addEvent("session.error", { message: String(error) });
  }
}

async function recordSessionItems(items) {
  if (!state.config?.session.managed || !items.length) return;
  try {
    const sessionId = await ensureManagedSession();
    const response = await fetch("/api/demo/session/items", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    const result = await jsonResponse(response);
    addEvent("session.items.append", result);
    await refreshSession();
  } catch (error) {
    stateMessage(elements.sessionItems, error instanceof Error ? error.message : String(error), "error");
    addEvent("session.error", { message: String(error) });
  }
}

async function listMemoryEntries() {
  if (!state.config?.memory.enabled) return;
  stateMessage(elements.memoryResults, "Loading memory entries…", "loading");
  try {
    const response = await fetch("/api/demo/memory/entries", { cache: "no-store" });
    const result = await jsonResponse(response);
    renderMemoryEntries(memoryEntries(result), "No memory entries for this actor yet.");
    addEvent("memory.entries.list", result);
  } catch (error) {
    stateMessage(elements.memoryResults, error instanceof Error ? error.message : String(error), "error");
    addEvent("memory.error", { message: String(error) });
  }
}

async function saveMemoryEntry() {
  const path = elements.memoryPath.value.trim();
  const content = elements.memoryFact.value.trim();
  if (!path || !content) {
    (path ? elements.memoryFact : elements.memoryPath).focus();
    return;
  }
  stateMessage(elements.memoryResults, "Saving memory entry…", "loading");
  try {
    const response = await fetch("/api/demo/memory/entries", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, content }),
    });
    const result = await jsonResponse(response);
    addEvent("memory.entry.create", result);
    elements.memoryHelp.textContent = `Saved ${path} for actor ${state.config.memory.actor}.`;
    await listMemoryEntries();
  } catch (error) {
    stateMessage(elements.memoryResults, error instanceof Error ? error.message : String(error), "error");
    addEvent("memory.error", { message: String(error) });
  }
}

async function searchMemoryEntries() {
  const query = elements.memoryQuery.value.trim();
  if (!query) {
    elements.memoryQuery.focus();
    return;
  }
  stateMessage(elements.memoryResults, "Searching memory…", "loading");
  try {
    const response = await fetch("/api/demo/memory/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit: 10 }),
    });
    const result = await jsonResponse(response);
    renderMemoryEntries(memoryEntries(result));
    addEvent("memory.entries.search", result);
  } catch (error) {
    stateMessage(elements.memoryResults, error instanceof Error ? error.message : String(error), "error");
    addEvent("memory.error", { message: String(error) });
  }
}

async function invokeSync(payload) {
  const response = await fetch("/invocations", {
    method: "POST",
    credentials: "same-origin",
    headers: invocationHeaders(),
    body: JSON.stringify(invocationPayload(payload)),
  });
  const result = await jsonResponse(response);
  addEvent("response", result);
  if (result.session_id) setSessionId(result.session_id);
  handleOutput(result.output);
  return result;
}

function parseSseFrame(frame) {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (!data || data === "[DONE]") return null;
  return JSON.parse(data);
}

async function invokeStreaming(payload) {
  const response = await fetch("/invocations", {
    method: "POST",
    credentials: "same-origin",
    headers: invocationHeaders(),
    body: JSON.stringify(invocationPayload({ ...payload, stream: true })),
  });
  if (!response.ok || !response.body) await jsonResponse(response);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const event = parseSseFrame(frame);
      if (event) handleEvent(event);
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const event = parseSseFrame(buffer);
    if (event) handleEvent(event);
  }
  finishDraft();
  return { status: state.pendingInterrupt ? "interrupted" : "completed" };
}

async function pollBackground(invocationId) {
  const deadline = Date.now() + 180000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 850));
    const response = await fetch(`/invocations/${encodeURIComponent(invocationId)}`, {
      cache: "no-store",
      credentials: "same-origin",
    });
    const result = await jsonResponse(response);
    addEvent("background.poll", result);
    if (result.status === "completed") {
      if (result.session_id) setSessionId(result.session_id);
      handleOutput(result.output);
      return result;
    }
    if (result.status === "failed") throw new Error(result.error || "Background invocation failed");
    setStatus(`Background · ${result.status}`, "busy");
  }
  throw new Error("Background invocation did not finish within three minutes.");
}

async function invokeBackground(payload) {
  const response = await fetch("/invocations", {
    method: "POST",
    credentials: "same-origin",
    headers: invocationHeaders(),
    body: JSON.stringify(invocationPayload({ ...payload, background: true })),
  });
  const started = await jsonResponse(response);
  addEvent("background.started", started);
  setStatus(`Background · ${started.id}`, "busy");
  return pollBackground(started.id);
}

async function dispatch(payload, mode = state.mode) {
  state.lastAssistantText = "";
  state.pendingInterrupt = null;
  elements.approvalPanel.hidden = true;
  if (mode === "streaming") return invokeStreaming(payload);
  if (mode === "background") return invokeBackground(payload);
  return invokeSync(payload);
}

async function sendText(text, mode = state.mode) {
  const content = text.trim();
  if (!content || state.busy) return "";
  appendMessage("user", content, "You");
  setBusy(true, mode === "background" ? "Starting background run" : mode === "streaming" ? "Streaming" : "Running");
  try {
    await dispatch({ input: [{ role: "user", content }] }, mode);
    const items = [{ role: "user", content, transport: mode, instance_id: state.instanceId }];
    if (state.lastAssistantText) {
      items.push({
        role: "assistant",
        content: state.lastAssistantText,
        transport: mode,
        instance_id: state.instanceId,
      });
    }
    await recordSessionItems(items);
    setConnection("online", "Connected");
    return state.lastAssistantText;
  } catch (error) {
    finishDraft();
    appendError(error);
    throw error;
  } finally {
    setBusy(false);
  }
}

async function resume(decision) {
  if (state.busy) return;
  if (!state.pendingInterrupt && !state.config?.session.durable) {
    appendError(new Error("No paused run is loaded for the current routing session."));
    return;
  }
  const payload =
    decision === "approve"
      ? { resume: { decisions: [{ type: "approve" }] } }
      : { resume: { decisions: [{ type: "reject", message: "Rejected from the Mason demo UI." }] } };
  appendMessage("system", decision === "approve" ? "Approved pending tool call." : "Rejected pending tool call.", "Human decision");
  setBusy(true, "Resuming");
  try {
    await dispatch(payload, "streaming");
    const items = [
      { role: "human_decision", content: decision, instance_id: state.instanceId },
    ];
    if (state.lastAssistantText) {
      items.push({
        role: "assistant",
        content: state.lastAssistantText,
        transport: "streaming",
        instance_id: state.instanceId,
      });
    }
    await recordSessionItems(items);
  } catch (error) {
    appendError(error);
  } finally {
    setBusy(false);
  }
}

function clearChat() {
  state.pendingInterrupt = null;
  state.draft = null;
  state.draftText = "";
  elements.approvalPanel.hidden = true;
  elements.chatLog.replaceChildren(elements.emptyState);
  elements.emptyState.hidden = false;
  elements.promptInput.focus();
  addEvent("chat.cleared", { session_id: state.sessionId });
}

async function loadConfig() {
  setConnection("loading", "Connecting");
  try {
    const response = await fetch("/api/demo/config", { cache: "no-store" });
    const config = await jsonResponse(response);
    state.config = config;
    state.instanceId = config.instance_id;
    setSessionId(config.session_id);
    elements.environmentBadge.textContent = config.app_control.restart_managed ? "Databricks App" : "Local runtime";
    elements.viewerValue.textContent = config.viewer;
    elements.identityValue.textContent = config.execution_identity;
    elements.identitySummary.textContent = `Agent executes as ${config.execution_identity}`;
    elements.streamingMode.textContent = config.streaming.transport;
    elements.backgroundMode.textContent = config.background.durable ? "Durable run store" : "In-process run store";
    elements.sessionMode.textContent = config.session.mode;
    elements.memoryMode.textContent = config.memory.enabled ? `Managed · actor ${config.memory.actor}` : "Not connected";
    elements.durabilityMode.textContent = config.durability.mode;
    elements.heartbeatMode.textContent = config.heartbeat.enabled
      ? `Every ${config.heartbeat.interval_seconds}s · stale after ${config.heartbeat.stale_after_seconds}s`
      : "Not connected";
    setCapability(elements.streamingStatus, config.streaming.enabled);
    setCapability(elements.backgroundStatus, config.background.enabled);
    setCapability(elements.sessionStatus, config.session.history);
    setCapability(elements.memoryStatus, config.memory.enabled);
    setCapability(elements.durabilityStatus, config.durability.enabled);
    setCapability(elements.heartbeatStatus, config.heartbeat.enabled);
    elements.rememberButton.disabled = state.busy || !config.memory.enabled;
    elements.searchMemory.disabled = state.busy || !config.memory.enabled;
    elements.askMemory.disabled = state.busy || !config.memory.enabled;
    elements.refreshSession.disabled = state.busy || !config.session.history;
    elements.resumeSession.disabled = state.busy || !config.session.durable;
    elements.rejectSession.disabled = state.busy || !config.session.durable;
    elements.memoryHelp.textContent = config.memory.enabled
      ? `${config.memory.store} · actor ${config.memory.actor}`
      : "Deploy with --with-memory-store to expose managed entries and agent memory tools.";
    elements.sessionStoreLabel.textContent = config.session.managed
      ? `${config.session.store} · actor ${config.session.actor} · the Apps routing cookie keys transcript and checkpoint state.`
      : config.session.history
        ? "Messages load from the in-process LangGraph checkpoint for the current routing cookie."
        : "Session history is unavailable.";
    updateRecoveryControls();
    elements.recoveryStatus.textContent = !config.session.durable
      ? "Stop is included, but durability needs a managed Session Store."
      : `Ready: ${config.recovery.steps.length} steps, about ${config.recovery.step_seconds}s each.`;
    setConnection("online", "Connected");
    addEvent("runtime.config", config);
    void refreshSession({ hydrateChat: true });
    if (config.recovery.enabled) {
      const recoverySessionId = state.recoverySessionId || state.sessionId;
      void monitorRecovery(recoverySessionId, { autoResume: true });
    }
    if (config.memory.enabled) void listMemoryEntries();
    else stateMessage(elements.memoryResults, "Connect a Memory Store to manage entries.");
    return config;
  } catch (error) {
    setConnection("offline", "Unavailable");
    elements.environmentBadge.textContent = "Runtime unavailable";
    throw error;
  }
}

function recoveryStatusText(result) {
  if (result.status === "failed") return `Sequence failed: ${result.error || "unknown error"}`;
  if (result.status === "completed") {
    const restored = result.outputs.some(
      (output) => output.instance_id && output.instance_id !== result.instance_id,
    );
    return restored
      ? "Recovered and completed. Earlier tool outputs were restored and skipped on this process."
      : "All tool steps completed for this routing session.";
  }
  if (result.needs_resume) {
    return `Heartbeat is stale. Starting attempt ${result.attempt + 1} from the first incomplete checkpoint…`;
  }
  if (result.owner_active && !result.worker_active) {
    const age = Number(result.heartbeat_age_seconds || 0).toFixed(1);
    return `Previous owner heartbeat is ${age}s old. Waiting until ${result.stale_after_seconds}s before takeover…`;
  }
  if (result.worker_active && result.outputs.length >= 2) {
    return `${result.outputs.length} steps are checkpointed. Stop App now to test recovery.`;
  }
  if (result.worker_active) return `Running ${result.current_step || "the next tool"}…`;
  return "Start App, wait for a few completed outputs, then stop the app process.";
}

function renderRecovery(result) {
  state.recovery = result;
  const outputs = new Map((result.outputs || []).map((output) => [output.tool, output]));
  elements.recoverySteps.replaceChildren();
  for (const [index, name] of (result.steps || []).entries()) {
    const output = outputs.get(name);
    const row = document.createElement("div");
    const restored = Boolean(output?.instance_id && output.instance_id !== result.instance_id);
    row.className = [
      "recovery-step",
      output ? "completed" : "pending",
      result.current_step === name ? "current" : "",
      restored ? "restored" : "",
    ].filter(Boolean).join(" ");

    const number = document.createElement("span");
    number.className = "recovery-step-index";
    number.textContent = output ? "✓" : String(index + 1);

    const body = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = name;
    const detail = document.createElement("small");
    if (output) {
      const process = output.instance_id ? ` · process ${output.instance_id}` : "";
      detail.textContent = `${output.output || "Completed"}${process}`;
    } else if (result.current_step === name) {
      detail.textContent = result.worker_active
        ? `Running on process ${result.instance_id}`
        : result.owner_active
          ? `Owned by process ${result.owner_id}`
          : "Will resume from this checkpoint";
    } else {
      detail.textContent = "Waiting";
    }
    body.append(title, detail);
    row.append(number, body);
    elements.recoverySteps.append(row);
  }

  elements.recoveryStatus.textContent = recoveryStatusText(result);
  elements.durabilityExecution.textContent = result.execution_id || "Not started";
  elements.durabilityExecution.title = result.execution_id || "";
  elements.durabilityAttempt.textContent = result.attempt || "—";
  elements.durabilityOwner.textContent = result.owner_id || "—";
  elements.durabilityOwner.title = result.owner_id || "";
  elements.durabilityHeartbeat.textContent = result.heartbeat_at
    ? `${Number(result.heartbeat_age_seconds || 0).toFixed(1)}s ago · ${result.heartbeat_fresh ? "fresh" : "stale"}`
    : "Not started";

  if (result.attempt && result.attempt !== state.lastDurabilityAttempt) {
    state.lastDurabilityAttempt = result.attempt;
    addEvent("durability.attempt", {
      execution_id: result.execution_id,
      attempt: result.attempt,
      owner_id: result.owner_id,
      atomic_claim: result.atomic_claim,
      claim_mode: result.claim_mode,
    });
  }
  if (result.heartbeat_at && result.heartbeat_at !== state.lastDurabilityHeartbeat) {
    state.lastDurabilityHeartbeat = result.heartbeat_at;
    addEvent("durability.heartbeat", {
      execution_id: result.execution_id,
      attempt: result.attempt,
      owner_id: result.owner_id,
      heartbeat_at: result.heartbeat_at,
      stale_after_seconds: result.stale_after_seconds,
    });
  }
  if (result.status === "stopped" && state.lastRecoveryStatus !== "stopped") {
    addEvent("durability.stale", {
      execution_id: result.execution_id,
      attempt: result.attempt,
      heartbeat_age_seconds: result.heartbeat_age_seconds,
    });
  }
  state.lastRecoveryStatus = result.status;
  const signature = JSON.stringify([
    result.status,
    result.attempt,
    result.current_step,
    result.worker_active,
    result.needs_resume,
    result.outputs,
  ]);
  if (signature !== state.recoverySignature) {
    state.recoverySignature = signature;
    addEvent("recovery.status", result);
  }
  updateRecoveryControls();
}

function scheduleRecoveryPoll(sessionId, delay = 1000) {
  if (state.recoveryPollTimer) clearTimeout(state.recoveryPollTimer);
  state.recoveryPollTimer = setTimeout(() => {
    monitorRecovery(sessionId, { autoResume: true }).catch(() => {});
  }, delay);
}

async function resumeRecoverySequence(sessionId) {
  const response = await fetch("/api/demo/app/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const result = await jsonResponse(response);
  renderRecovery(result);
  addEvent("app.start", result);
  scheduleRecoveryPoll(sessionId);
  return result;
}

async function monitorRecovery(sessionId, { autoResume = false } = {}) {
  if (!state.config?.recovery.enabled || !sessionId) return null;
  state.recoverySessionId = sessionId;
  try {
    const response = await fetch("/api/demo/recovery", {
      cache: "no-store",
      credentials: "same-origin",
    });
    const result = await jsonResponse(response);
    renderRecovery(result);
    setConnection("online", "Connected");
    if (result.needs_resume && autoResume) return resumeRecoverySequence(sessionId);
    if (result.status === "running" || result.needs_resume) scheduleRecoveryPoll(sessionId);
    return result;
  } catch (error) {
    if (recoveryInProgress()) {
      setConnection("offline", "Process stopped");
      elements.recoveryStatus.textContent = "Waiting for the app process to restart…";
      scheduleRecoveryPoll(sessionId, 1500);
      return null;
    }
    throw error;
  }
}

async function startApp() {
  if (state.busy || !state.config?.recovery.enabled) return;
  const sessionId = ensureSessionId();
  state.recoverySessionId = sessionId;
  elements.startApp.disabled = true;
  elements.recoveryStatus.textContent = "Starting durable app work…";
  try {
    const response = await fetch("/api/demo/app/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const result = await jsonResponse(response);
    renderRecovery(result);
    addEvent("app.start", result);
    scheduleRecoveryPoll(sessionId);
  } catch (error) {
    elements.recoveryStatus.textContent = error instanceof Error ? error.message : String(error);
    appendError(error);
    updateRecoveryControls();
  }
}

async function waitForRestart(previousInstanceId) {
  const deadline = Date.now() + 180000;
  elements.recoveryStatus.textContent = "Waiting for a new app process…";
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    try {
      const response = await fetch("/api/demo/config", { cache: "no-store" });
      if (!response.ok) continue;
      const config = await response.json();
      if (config.instance_id && config.instance_id !== previousInstanceId) {
        state.config = config;
        state.instanceId = config.instance_id;
        setConnection("online", "Restarted");
        addEvent("runtime.restarted", config);
        return config;
      }
      setConnection("loading", "Restarting");
    } catch {
      setConnection("offline", "Process stopped");
    }
  }
  throw new Error("The app did not restart within three minutes. Local runs need an auto-restarting supervisor.");
}

async function stopCurrentApp() {
  const previousInstanceId = state.instanceId;
  setBusy(true, "Stopping");
  elements.recoveryStatus.textContent = "Stopping the current app process…";
  try {
    const response = await fetch("/api/demo/app/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const result = await jsonResponse(response);
    addEvent("app.stop", result);
    await waitForRestart(previousInstanceId);
    await loadConfig();
  } finally {
    setBusy(false);
  }
}

async function stopAndRecover() {
  if (state.busy || !state.config?.app_control.stop_enabled) return;
  if (state.pendingInterrupt) {
    try {
      await stopCurrentApp();
      elements.approvalPanel.hidden = false;
      elements.recoveryStatus.textContent = "New process is ready. Approve or reject the same paused run.";
    } catch (error) {
      elements.recoveryStatus.textContent = error instanceof Error ? error.message : String(error);
      appendError(error);
    }
    return;
  }
  if (!state.recovery?.worker_active || !state.recoverySessionId) {
    appendError(new Error("Start App before stopping the app process."));
    return;
  }
  const sessionId = state.recoverySessionId;
  try {
    await stopCurrentApp();
    elements.recoveryStatus.textContent = "Host restarted. Waiting for stale-heartbeat takeover…";
    await monitorRecovery(sessionId, { autoResume: true });
  } catch (error) {
    elements.recoveryStatus.textContent = error instanceof Error ? error.message : String(error);
    if (!String(error).includes("Failed to fetch")) appendError(error);
  }
}

elements.composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = elements.promptInput.value;
  if (!text.trim()) return;
  elements.promptInput.value = "";
  elements.promptInput.style.height = "auto";
  try {
    await sendText(text);
  } catch {
    elements.promptInput.value = text;
  }
});

elements.promptInput.addEventListener("input", () => {
  elements.promptInput.style.height = "auto";
  elements.promptInput.style.height = `${Math.min(elements.promptInput.scrollHeight, 180)}px`;
});

elements.promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    document.querySelectorAll(".mode-button").forEach((item) => item.classList.toggle("active", item === button));
  });
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.promptInput.value = button.dataset.prompt;
    elements.promptInput.dispatchEvent(new Event("input"));
    elements.promptInput.focus();
  });
});

elements.copySession.addEventListener("click", async () => {
  await navigator.clipboard.writeText(ensureSessionId());
  const label = elements.copySession.querySelector("span");
  label.textContent = "Copied";
  setTimeout(() => { label.textContent = "Copy"; }, 1200);
});

elements.refreshConfig.addEventListener("click", () => loadConfig().catch(appendError));
elements.refreshSession.addEventListener("click", () => refreshSession({ hydrateChat: true }));
elements.clearEvents.addEventListener("click", () => {
  state.events = [];
  elements.eventLog.innerHTML = '<div class="event-empty">Invocation events appear here.</div>';
});
elements.approveAction.addEventListener("click", () => resume("approve"));
elements.rejectAction.addEventListener("click", () => resume("reject"));
elements.resumeSession.addEventListener("click", () => resume("approve"));
elements.rejectSession.addEventListener("click", () => resume("reject"));

elements.rememberButton.addEventListener("click", saveMemoryEntry);
elements.searchMemory.addEventListener("click", searchMemoryEntries);
elements.askMemory.addEventListener("click", async () => {
  const query = elements.memoryQuery.value.trim() || "my saved profile and preferences";
  clearChat();
  await sendText(`Use the recall tool to find what you remember about ${query}.`, "sync").catch(() => {});
});

elements.startApp.addEventListener("click", startApp);
elements.stopApp.addEventListener("click", stopAndRecover);

loadConfig().catch((error) => {
  appendError(error);
  elements.memoryHelp.textContent = "Runtime configuration is unavailable.";
  elements.recoveryStatus.textContent = "Runtime configuration is unavailable.";
});
