const elements = {
  approvalPanel: document.querySelector("#approval-panel"),
  approvalSummary: document.querySelector("#approval-summary"),
  approveAction: document.querySelector("#approve-action"),
  backgroundMode: document.querySelector("#background-mode-value"),
  backgroundStatus: document.querySelector("#background-status"),
  chatLog: document.querySelector("#chat-log"),
  clearEvents: document.querySelector("#clear-events"),
  composer: document.querySelector("#composer"),
  copySession: document.querySelector("#copy-session"),
  emptyState: document.querySelector("#empty-state"),
  eventLog: document.querySelector("#event-log"),
  memoryHelp: document.querySelector("#memory-help"),
  memoryMode: document.querySelector("#memory-mode-value"),
  memoryResults: document.querySelector("#memory-results"),
  memoryStatus: document.querySelector("#memory-status"),
  refreshMemory: document.querySelector("#refresh-memory"),
  newSession: document.querySelector("#new-session"),
  promptInput: document.querySelector("#prompt-input"),
  refreshConfig: document.querySelector("#refresh-config"),
  refreshSession: document.querySelector("#refresh-session"),
  rejectAction: document.querySelector("#reject-action"),
  rejectSession: document.querySelector("#reject-session"),
  resumeSession: document.querySelector("#resume-session"),
  runStatus: document.querySelector("#run-status"),
  sendButton: document.querySelector("#send-button"),
  sessionId: document.querySelector("#session-id"),
  sessionItems: document.querySelector("#session-items"),
  sessionList: document.querySelector("#session-list"),
  sessionMode: document.querySelector("#session-mode-value"),
  sessionStatus: document.querySelector("#session-status"),
  sessionStoreLabel: document.querySelector("#session-store-label"),
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
  lastAssistantText: "",
  managedSessionId: "",
  mode: "streaming",
  pendingInterrupt: null,
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
  elements.refreshMemory.disabled = busy || !state.config?.memory.enabled;
  elements.newSession.disabled = busy;
  elements.refreshSession.disabled = busy || !state.config?.session.history;
  elements.resumeSession.disabled = busy || !state.config?.session.durable;
  elements.rejectSession.disabled = busy || !state.config?.session.durable;
  document.querySelectorAll(".session-open-button").forEach((button) => {
    button.disabled = busy;
  });
  if (busy) setStatus(label, "busy");
  else if (!elements.runStatus.classList.contains("error")) setStatus("Ready");
}

// Human-readable name for each orb color, surfaced as a hover tooltip so the
// green/amber/red states are self-explanatory.
const CAPABILITY_SUMMARY = { enabled: "Available", degraded: "Limited", disabled: "Unavailable" };

function setCapability(element, state, detail, command) {
  const key = state === true ? "enabled" : state === "degraded" ? "degraded" : "disabled";
  element.classList.toggle("enabled", key === "enabled");
  element.classList.toggle("degraded", key === "degraded");
  element.classList.toggle("disabled", key === "disabled");
  element.setAttribute("role", "img");
  element.setAttribute("aria-label", `Status: ${CAPABILITY_SUMMARY[key]}`);
  const row = element.closest(".capability-row");
  if (!row) return;

  let tooltip = row.querySelector(".capability-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("span");
    tooltip.id = `${element.id}-tooltip`;
    tooltip.className = "capability-tooltip";
    tooltip.setAttribute("role", "tooltip");
    row.append(tooltip);
    row.tabIndex = 0;
    row.setAttribute("aria-describedby", tooltip.id);
  }
  tooltip.dataset.state = key;
  tooltip.replaceChildren();

  const summary = document.createElement("strong");
  summary.className = "capability-tooltip-status";
  summary.textContent = CAPABILITY_SUMMARY[key];
  tooltip.append(summary);

  if (detail) {
    const description = document.createElement("span");
    description.className = "capability-tooltip-detail";
    description.textContent = detail;
    tooltip.append(description);
  }
  if (command) {
    const guidance = document.createElement("span");
    guidance.className = "capability-tooltip-guidance";
    guidance.textContent = "Connect by redeploying:";
    const commandText = document.createElement("code");
    commandText.className = "capability-tooltip-command";
    commandText.textContent = command;
    tooltip.append(guidance, commandText);
  }
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

function sessions(payload) {
  return payload?.sessions || [];
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

function renderSessions(items) {
  elements.sessionList.replaceChildren();
  if (!items.length) {
    stateMessage(elements.sessionList, "No sessions yet.");
    return;
  }
  for (const session of items) {
    const current = session.session_id === state.sessionId;
    const item = document.createElement("article");
    item.className = `state-item session-item${current ? " current" : ""}`;
    const heading = document.createElement("div");
    heading.className = "session-item-heading";
    const title = document.createElement("strong");
    title.textContent = current ? "Current session" : "Session";
    heading.append(title);
    if (!current && state.config?.session.managed) {
      const open = document.createElement("button");
      open.className = "text-button session-open-button";
      open.type = "button";
      open.textContent = "Open";
      open.disabled = state.busy;
      open.addEventListener("click", () => openSession(session.session_id));
      heading.append(open);
    }
    const content = document.createElement("p");
    content.textContent = session.session_id || "Unknown session";
    const meta = document.createElement("small");
    meta.textContent = [
      session.actor_id,
      session.last_activity_time || session.create_time,
      current ? "active routing cookie" : "",
    ].filter(Boolean).join(" · ");
    item.append(heading, content, meta);
    elements.sessionList.append(item);
  }
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

async function refreshSessions() {
  stateMessage(elements.sessionList, "Loading sessions…", "loading");
  try {
    const response = await fetch("/api/demo/sessions", { cache: "no-store" });
    const result = await jsonResponse(response);
    renderSessions(sessions(result));
    addEvent("sessions.list", result);
  } catch (error) {
    stateMessage(elements.sessionList, error instanceof Error ? error.message : String(error), "error");
    addEvent("session.error", { message: String(error) });
  }
}

async function refreshSessionView({ hydrateChat = false } = {}) {
  await refreshSession({ hydrateChat });
  await refreshSessions();
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
    await refreshSessionView();
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

function clearChat({ recordEvent = true } = {}) {
  state.pendingInterrupt = null;
  state.draft = null;
  state.draftText = "";
  state.lastAssistantText = "";
  elements.approvalPanel.hidden = true;
  elements.chatLog.replaceChildren(elements.emptyState);
  elements.emptyState.hidden = false;
  elements.promptInput.focus();
  if (recordEvent) addEvent("chat.cleared", { session_id: state.sessionId });
}

function resetSessionState() {
  clearChat({ recordEvent: false });
  stateMessage(elements.sessionList, "Loading sessions…", "loading");
  stateMessage(elements.sessionItems, "Loading transcript…", "loading");
}

async function createNewSession() {
  if (state.busy) return;
  setBusy(true, "Creating session");
  try {
    const response = await fetch("/api/session/new", {
      method: "POST",
      credentials: "same-origin",
    });
    const result = await jsonResponse(response);
    setSessionId(result.session_id);
    resetSessionState();
    addEvent("session.new", result);
    if (state.config?.session.managed) await ensureManagedSession();
    await refreshSessionView({ hydrateChat: true });
  } catch (error) {
    appendError(error);
  } finally {
    setBusy(false);
  }
}

async function openSession(sessionId) {
  if (state.busy || !sessionId || sessionId === state.sessionId) return;
  setBusy(true, "Opening session");
  try {
    const response = await fetch(`/api/demo/sessions/${encodeURIComponent(sessionId)}/open`, {
      method: "POST",
      credentials: "same-origin",
    });
    const result = await jsonResponse(response);
    setSessionId(result.session_id);
    resetSessionState();
    addEvent("session.open", result);
    await refreshSessionView({ hydrateChat: true });
  } catch (error) {
    appendError(error);
  } finally {
    setBusy(false);
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/demo/config", { cache: "no-store" });
    const config = await jsonResponse(response);
    state.config = config;
    state.instanceId = config.instance_id;
    setSessionId(config.session_id);
    elements.viewerValue.textContent = config.viewer;
    elements.streamingMode.textContent = config.streaming.transport;
    elements.backgroundMode.textContent = config.background.durable ? "Durable run store" : "In-process run store";
    elements.sessionMode.textContent = config.session.mode;
    elements.memoryMode.textContent = config.memory.enabled ? `Managed · actor ${config.memory.actor}` : "Not connected";
    setCapability(
      elements.streamingStatus,
      config.streaming.enabled,
      config.streaming.enabled
        ? `Streaming responses over ${config.streaming.transport}.`
        : "Streaming is disabled for this deployment.",
    );
    setCapability(
      elements.backgroundStatus,
      config.background.enabled,
      config.background.enabled
        ? `Background invocations use a ${config.background.durable ? "durable" : "in-process"} run store.`
        : "Background invocations are disabled.",
    );
    setCapability(
      elements.sessionStatus,
      config.session.managed ? true : config.session.history ? "degraded" : false,
      config.session.managed
        ? `Connected to Session Store "${config.session.store}" for actor ${config.session.actor}. State is durable and shareable.`
        : "Session state is kept in process and resets when the app restarts.",
      config.session.managed ? null : "mason deploy <deployment-name> --session <store-name>",
    );
    setCapability(
      elements.memoryStatus,
      config.memory.enabled,
      config.memory.enabled
        ? `Connected to ${config.memory.store} for actor ${config.memory.actor}.`
        : "No long-term memory is connected.",
      config.memory.enabled ? null : "mason deploy <deployment-name> --memory <store-name>",
    );
    elements.refreshMemory.disabled = state.busy || !config.memory.enabled;
    elements.newSession.disabled = state.busy;
    elements.refreshSession.disabled = state.busy || !config.session.history;
    elements.resumeSession.disabled = state.busy || !config.session.durable;
    elements.rejectSession.disabled = state.busy || !config.session.durable;
    elements.memoryHelp.textContent = config.memory.enabled
      ? `${config.memory.store} · actor ${config.memory.actor}`
      : "Deploy with --memory to expose managed entries and agent memory tools.";
    elements.sessionStoreLabel.textContent = config.session.managed
      ? `${config.session.store} · actor ${config.session.actor} · the Apps routing cookie keys transcript and checkpoint state.`
      : config.session.history
        ? "Messages load from the in-process LangGraph checkpoint for the current routing cookie."
        : "Session history is unavailable.";
    addEvent("runtime.config", config);
    void refreshSessionView({ hydrateChat: true });
    if (config.memory.enabled) void listMemoryEntries();
    else stateMessage(elements.memoryResults, "Connect a Memory Store to manage entries.");
    return config;
  } catch (error) {
    throw error;
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
elements.newSession.addEventListener("click", createNewSession);
elements.refreshSession.addEventListener("click", () => refreshSessionView({ hydrateChat: true }));
elements.clearEvents.addEventListener("click", () => {
  state.events = [];
  elements.eventLog.innerHTML = '<div class="event-empty">Invocation events appear here.</div>';
});
elements.approveAction.addEventListener("click", () => resume("approve"));
elements.rejectAction.addEventListener("click", () => resume("reject"));
elements.resumeSession.addEventListener("click", () => resume("approve"));
elements.rejectSession.addEventListener("click", () => resume("reject"));

elements.refreshMemory.addEventListener("click", listMemoryEntries);

loadConfig().catch((error) => {
  appendError(error);
  elements.memoryHelp.textContent = "Runtime configuration is unavailable.";
});
