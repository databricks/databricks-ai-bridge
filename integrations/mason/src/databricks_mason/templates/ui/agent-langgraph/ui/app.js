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
  memoryMode: document.querySelector("#memory-mode-value"),
  memoryStatus: document.querySelector("#memory-status"),
  tracingMode: document.querySelector("#tracing-mode-value"),
  tracingStatus: document.querySelector("#tracing-status"),
  modelSelect: document.querySelector("#model-select"),
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
  viewerAvatar: document.querySelector("#viewer-avatar"),
  brandSub: document.querySelector("#brand-sub"),
  sessionSearch: document.querySelector("#session-search-input"),
  eventFilters: document.querySelector("#event-filters"),
  eventDetail: document.querySelector("#event-detail"),
  eventDetailJson: document.querySelector("#event-detail-json"),
  copyEventDetail: document.querySelector("#copy-event-detail"),
  eventViewLogs: document.querySelector("#event-view-logs"),
  eventViewList: document.querySelector("#event-view-list"),
  leftCard: document.querySelector(".left-card"),
  capabilitiesBlock: document.querySelector("#capabilities-block"),
  memoryPane: document.querySelector("#memory-pane"),
  memoryBack: document.querySelector("#memory-back"),
  memoryRefresh: document.querySelector("#memory-refresh"),
  memoryStoreName: document.querySelector("#memory-store-name"),
  memoryStoreResource: document.querySelector("#memory-store-resource"),
  memoryActor: document.querySelector("#memory-actor-input"),
  memorySearch: document.querySelector("#memory-search-input"),
  memoryCount: document.querySelector("#memory-count"),
  memoryEntries: document.querySelector("#memory-entries"),
  memoryModal: document.querySelector("#memory-modal"),
  memoryModalTitle: document.querySelector("#memory-modal-title"),
  memoryModalBody: document.querySelector("#memory-modal-body"),
  memoryModalFoot: document.querySelector("#memory-modal-foot"),
  memoryModalClose: document.querySelector("#memory-modal-close"),
};

// The framework label shown in the header subtitle, next to the host.
const AGENT_FRAMEWORK = "LangGraph";

const SESSION_STORAGE_KEY = "databricks-mason-session-id";

function newSessionId() {
  return crypto.randomUUID();
}

const state = {
  busy: false,
  config: null,
  draft: null,
  draftText: "",
  events: [],
  eventSeq: 0,
  eventView: "logs",
  eventFilter: "all",
  selectedEventId: null,
  sessionFilter: "",
  sessions: [],
  instanceId: null,
  lastAssistantText: "",
  managedSessionId: "",
  model: "",
  mode: "streaming",
  pendingInterrupt: null,
  sessionId: localStorage.getItem(SESSION_STORAGE_KEY) || newSessionId(),
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
  localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);
  elements.sessionId.textContent = state.sessionId;
}

function demoUrl(path) {
  const url = new URL(path, window.location.origin);
  url.searchParams.set("session_id", ensureSessionId());
  return `${url.pathname}${url.search}`;
}

function setStatus(label, type = "ready") {
  elements.runStatus.textContent = label;
  elements.runStatus.className = `run-status ${type === "ready" ? "" : type}`.trim();
}

// The send button is enabled only when there is text to send and no run is in flight.
function updateSendState() {
  elements.sendButton.disabled = state.busy || !elements.promptInput.value.trim();
}

function setBusy(busy, label = "Working") {
  state.busy = busy;
  elements.chatLog.setAttribute("aria-busy", String(busy));
  updateSendState();
  elements.promptInput.disabled = busy;
  elements.approveAction.disabled = busy;
  elements.rejectAction.disabled = busy;
  elements.newSession.disabled = busy;
  elements.modelSelect.disabled = busy || elements.modelSelect.options.length <= 1;
  elements.refreshSession.disabled = busy || !state.config?.session.history;
  elements.resumeSession.disabled = busy || !state.config?.session.durable;
  elements.rejectSession.disabled = busy || !state.config?.session.durable;
  document.querySelectorAll(".session-open-button").forEach((button) => {
    button.disabled = busy;
  });
  if (busy) setStatus(label, "busy");
  else if (!elements.runStatus.classList.contains("error")) setStatus("Ready");
}

// Human-readable name for each orb color, shown in the expanded capability detail.
const CAPABILITY_SUMMARY = { enabled: "Available", degraded: "Limited", disabled: "Unavailable" };

// Populate a capability's status orb and its click-to-expand detail panel.
// `link` (optional) renders an external link in the detail, e.g. to an MLflow experiment.
function setCapability(element, state, detail, command, link) {
  const key = state === true ? "enabled" : state === "degraded" ? "degraded" : "disabled";
  element.classList.toggle("enabled", key === "enabled");
  element.classList.toggle("degraded", key === "degraded");
  element.classList.toggle("disabled", key === "disabled");
  element.setAttribute("role", "img");
  element.setAttribute("aria-label", `Status: ${CAPABILITY_SUMMARY[key]}`);
  const item = element.closest(".capability-item");
  const panel = item?.querySelector(".capability-detail");
  if (!panel) return;

  panel.className = `capability-detail ${key}`;
  panel.hidden = !item.classList.contains("expanded");
  panel.replaceChildren();

  const status = document.createElement("span");
  status.className = "cap-status";
  status.textContent = CAPABILITY_SUMMARY[key];
  panel.append(status);

  if (detail) {
    const description = document.createElement("span");
    description.textContent = detail;
    panel.append(description);
  }
  if (command) {
    const label = document.createElement("span");
    label.className = "cap-command-label";
    label.textContent = "Connect by running:";
    const row = document.createElement("div");
    row.className = "cap-command";
    const code = document.createElement("code");
    code.textContent = command;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.setAttribute("aria-label", `Copy command: ${command}`);
    copy.innerHTML =
      '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    copy.addEventListener("click", (event) => {
      event.stopPropagation();
      void navigator.clipboard.writeText(command).catch(() => {});
    });
    row.append(code, copy);
    panel.append(label, row);
  }
  if (link) {
    const anchor = document.createElement("a");
    anchor.className = "cap-link";
    anchor.href = link.href;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    anchor.append(
      iconSvg(
        "icon",
        '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
      ),
    );
    const label = document.createElement("span");
    label.textContent = link.label;
    anchor.append(label);
    panel.append(anchor);
  }
}

function formatJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Basic JSON syntax highlighting: wraps tokens in <span> classes for coloring.
// The payload is HTML-escaped first, so runtime data can't inject markup.
function highlightJson(value) {
  const json = escapeHtml(formatJson(value));
  return json.replace(
    /("(?:\\.|[^"\\])*"(\s*:)?|\b(?:true|false)\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      let cls = "json-num";
      if (match.startsWith("&quot;") || match.startsWith('"')) {
        cls = /:\s*$/.test(match) ? "json-key" : "json-str";
      } else if (match === "true" || match === "false") {
        cls = "json-bool";
      } else if (match === "null") {
        cls = "json-null";
      }
      return `<span class="${cls}">${match}</span>`;
    },
  );
}

// Bucket an event type into one of the filter categories used by the list view.
function eventCategory(type) {
  const value = String(type || "").toLowerCase();
  if (value === "error" || value.endsWith(".error")) return "error";
  if (value.startsWith("session")) return "sessions";
  if (value.startsWith("memory")) return "memory";
  if (
    value.startsWith("background") ||
    value.startsWith("model") ||
    value.startsWith("run") ||
    ["response", "delta", "message", "interrupt", "runtime.config"].includes(value)
  ) {
    return "invocation";
  }
  return "system";
}

// A short one-line summary for the list view, derived from the payload.
function eventSummary(type, payload) {
  if (payload && typeof payload === "object") {
    for (const key of ["message", "summary", "detail", "status", "error"]) {
      if (typeof payload[key] === "string" && payload[key]) return payload[key];
    }
  }
  return String(type || "event");
}

function eventRunId(payload) {
  if (payload && typeof payload === "object") {
    return payload.run_id || payload.runId || payload.id || payload.invocation_id || "";
  }
  return "";
}

function addEvent(type, payload) {
  state.events.unshift({
    id: `evt-${(state.eventSeq += 1)}`,
    type,
    category: eventCategory(type),
    summary: eventSummary(type, payload),
    runId: eventRunId(payload),
    payload,
    at: new Date(),
  });
  state.events = state.events.slice(0, 80);
  renderEvents();
}

function eventEmpty() {
  const wrap = document.createElement("div");
  wrap.className = "event-empty";
  wrap.innerHTML =
    '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>' +
    "<strong>No events in this session</strong>" +
    "<span>Start an agent run to collect runtime evidence.</span>";
  return wrap;
}

function renderEventDetail() {
  const event = state.events.find((item) => item.id === state.selectedEventId);
  const show = state.eventView === "list" && Boolean(event);
  elements.eventDetail.hidden = !show;
  if (show) elements.eventDetailJson.innerHTML = highlightJson(event.payload);
}

function renderEvents() {
  const listView = state.eventView === "list";
  elements.eventFilters.hidden = !listView;
  elements.eventLog.classList.toggle("list-view", listView);
  elements.eventLog.classList.toggle("logs-view", !listView);
  elements.eventLog.replaceChildren();

  const events = listView
    ? state.events.filter((event) => state.eventFilter === "all" || event.category === state.eventFilter)
    : state.events;

  if (!events.length) {
    elements.eventLog.append(eventEmpty());
    renderEventDetail();
    return;
  }

  if (!listView) {
    for (const event of events) {
      const entry = document.createElement("div");
      entry.className = "event-entry";
      const header = document.createElement("div");
      header.className = "event-entry-header";
      const name = document.createElement("span");
      name.className = `event-type cat-${event.category}`;
      name.textContent = event.type;
      const time = document.createElement("span");
      time.className = "event-time";
      time.textContent = event.at.toLocaleTimeString();
      const body = document.createElement("pre");
      body.innerHTML = highlightJson(event.payload);
      header.append(name, time);
      entry.append(header, body);
      elements.eventLog.append(entry);
    }
    renderEventDetail();
    return;
  }

  for (const event of events) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `event-row${event.id === state.selectedEventId ? " selected" : ""}`;
    const top = document.createElement("div");
    top.className = "event-row-top";
    const type = document.createElement("span");
    type.className = `event-row-type cat-${event.category}`;
    type.textContent = event.type;
    const time = document.createElement("span");
    time.className = "event-row-time";
    time.textContent = event.at.toLocaleTimeString();
    top.append(type, time);
    const summary = document.createElement("div");
    summary.className = "event-row-summary";
    summary.textContent = event.summary;
    row.append(top, summary);
    if (event.runId) {
      const run = document.createElement("div");
      run.className = "event-row-run";
      run.textContent = event.runId;
      row.append(run);
    }
    row.addEventListener("click", () => {
      state.selectedEventId = event.id;
      renderEvents();
    });
    elements.eventLog.append(row);
  }
  renderEventDetail();
}

function setEventView(view) {
  state.eventView = view;
  elements.eventViewLogs.classList.toggle("active", view === "logs");
  elements.eventViewLogs.setAttribute("aria-pressed", String(view === "logs"));
  elements.eventViewList.classList.toggle("active", view === "list");
  elements.eventViewList.setAttribute("aria-pressed", String(view === "list"));
  renderEvents();
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

function initials(name) {
  const parts = String(name || "")
    .replace(/@.*/, "")
    .split(/[.\s_-]+/)
    .filter(Boolean);
  if (!parts.length) return "•";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function setViewer(name) {
  const label = name || "Local developer";
  elements.viewerValue.textContent = label;
  if (elements.viewerAvatar) elements.viewerAvatar.textContent = initials(label);
}

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

// Hover actions under an agent message. "View trace" is intentionally omitted
// until the demo backend exposes a trace URL (per design feedback).
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
      await navigator.clipboard.writeText(textEl.textContent || "");
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

function appendMessage(role, content, label, { time } = {}) {
  hideEmptyState();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  const text = document.createElement("div");
  text.className = "message-content";
  text.textContent = content;

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

function appendError(error) {
  const message = error instanceof Error ? error.message : String(error);
  appendMessage("error", message, "Request failed");
  setStatus("Error", "error");
  addEvent("error", { message });
}

function startDraft() {
  if (state.draft) return state.draft;
  const draft = appendMessage("assistant", "", "Agent");
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

function invocationPayload(payload, transport = {}) {
  const sessionId = ensureSessionId();
  const input = {
    session_id: sessionId,
    actor: state.config?.session.actor || sessionId,
    ...payload,
  };
  if (state.model) input.model = state.model;
  return { id: newSessionId(), input, ...transport };
}

function agentResult(result) {
  return result?.output && !Array.isArray(result.output) ? result.output : result;
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

// ---- Memory pane -----------------------------------------------------------

const memoryState = { query: "", searchTimer: null };

function prettyDate(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString([], { year: "numeric", month: "short", day: "numeric" });
}

function prettyDateTime(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function iconSvg(className, inner) {
  const span = document.createElement("span");
  span.innerHTML = `<svg class="${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`;
  return span.firstChild;
}

// Store name = the trailing segment of "memory-stores/<name>"; resource = the full path.
function memoryStoreName() {
  const store = state.config?.memory.store || "";
  const parts = store.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "Memory store";
}

function currentMemoryActor() {
  return (elements.memoryActor.value || "").trim() || state.config?.memory.actor || "";
}

function memoryMessage(text, kind = "empty") {
  elements.memoryEntries.replaceChildren();
  const div = document.createElement("div");
  div.className = `memory-${kind}`;
  div.textContent = text;
  elements.memoryEntries.append(div);
  elements.memoryCount.textContent = "";
}

function renderMemoryCards(entries, query) {
  elements.memoryEntries.replaceChildren();
  elements.memoryCount.textContent = entries.length
    ? query
      ? `${entries.length} matching`
      : `${entries.length} of ${entries.length} entries`
    : "";
  if (!entries.length) {
    memoryMessage(query ? "No entries match your search." : "No memory entries for this actor yet.");
    return;
  }
  for (const entry of entries) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "memory-entry";
    const top = document.createElement("div");
    top.className = "memory-entry-top";
    const path = document.createElement("span");
    path.className = "memory-entry-path";
    path.textContent = entry.path || entry.name || "Memory entry";
    top.append(path, iconSvg("memory-entry-expand icon", '<path d="M7 7h10v10"/><path d="M7 17 17 7"/>'));
    const preview = document.createElement("p");
    preview.className = "memory-entry-preview";
    preview.textContent = extractText(entry.content) || entry.description || "";
    const foot = document.createElement("div");
    foot.className = "memory-entry-foot";
    const actor = document.createElement("span");
    actor.className = "memory-entry-actor";
    actor.textContent = entry.actor_id || "";
    const time = document.createElement("span");
    time.className = "memory-entry-time";
    time.textContent = prettyDate(entry.update_time || entry.create_time);
    foot.append(actor, time);
    card.append(top, preview, foot);
    card.addEventListener("click", () => openMemoryModal(entry));
    elements.memoryEntries.append(card);
  }
}

async function loadMemory() {
  if (!state.config?.memory.enabled) {
    memoryMessage("Connect a Memory Store to browse entries: mason memory bind <store-name>");
    return;
  }
  const query = memoryState.query.trim();
  const actor = currentMemoryActor();
  memoryMessage(query ? "Searching…" : "Loading memory…", "loading");
  try {
    let entries;
    if (query) {
      const response = await fetch(demoUrl("/api/demo/memory/search"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 50, actor: actor || undefined }),
      });
      entries = memoryEntries(await jsonResponse(response));
    } else {
      const url = demoUrl("/api/demo/memory/entries") + (actor ? `&actor=${encodeURIComponent(actor)}` : "");
      const response = await fetch(url, { cache: "no-store" });
      entries = memoryEntries(await jsonResponse(response));
    }
    renderMemoryCards(entries, query);
    addEvent(query ? "memory.entries.search" : "memory.entries.list", {
      count: entries.length,
      actor,
      query: query || undefined,
    });
  } catch (error) {
    memoryMessage(error instanceof Error ? error.message : String(error), "error");
    addEvent("memory.error", { message: String(error) });
  }
}

function openMemoryPane() {
  elements.leftCard.classList.add("memory-open");
  elements.capabilitiesBlock.hidden = true;
  elements.memoryPane.hidden = false;
  elements.memoryStoreName.textContent = memoryStoreName();
  elements.memoryStoreResource.textContent = state.config?.memory.store || "";
  if (!elements.memoryActor.value) elements.memoryActor.value = state.config?.memory.actor || "";
  void loadMemory();
}

function closeMemoryPane() {
  elements.leftCard.classList.remove("memory-open");
  elements.memoryPane.hidden = true;
  elements.capabilitiesBlock.hidden = false;
}

function openMemoryModal(entry) {
  elements.memoryModalTitle.textContent = entry.path || entry.name || "Memory entry";
  elements.memoryModalBody.textContent = extractText(entry.content) || entry.description || "(no content)";
  elements.memoryModalFoot.textContent = [entry.actor_id, prettyDateTime(entry.update_time || entry.create_time)]
    .filter(Boolean)
    .join(" · ");
  elements.memoryModal.hidden = false;
}

function closeMemoryModal() {
  elements.memoryModal.hidden = true;
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

function relativeTime(value) {
  if (!value) return "";
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return "";
  const minutes = Math.round((Date.now() - then) / 60000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? "Yesterday" : `${days} days ago`;
}

function renderSessions(items) {
  state.sessions = items || [];
  renderSessionList();
}

function renderSessionList() {
  elements.sessionList.replaceChildren();
  if (!state.sessions.length) {
    stateMessage(elements.sessionList, "No sessions yet.");
    return;
  }
  const query = state.sessionFilter.trim().toLowerCase();
  const items = state.sessions.filter(
    (session) =>
      !query ||
      String(session.session_id || "").toLowerCase().includes(query) ||
      String(session.actor_id || "").toLowerCase().includes(query),
  );
  if (!items.length) {
    stateMessage(elements.sessionList, "No sessions match your search.");
    return;
  }
  for (const session of items) {
    const current = session.session_id === state.sessionId;
    const canOpen = !current && state.config?.session.managed;
    const row = document.createElement(canOpen ? "button" : "div");
    row.className = `session-item${current ? " current" : ""}${canOpen ? " session-open-button" : ""}`;
    if (canOpen) {
      row.type = "button";
      row.disabled = state.busy;
      row.addEventListener("click", () => openSession(session.session_id));
    }
    const top = document.createElement("div");
    top.className = "session-item-top";
    const title = document.createElement("span");
    title.className = "session-item-title";
    title.textContent = current ? "Current session" : "Session";
    const time = document.createElement("span");
    time.className = "session-item-time";
    time.textContent = current ? "Active" : relativeTime(session.last_activity_time || session.create_time);
    top.append(title, time);
    const id = document.createElement("span");
    id.className = "session-item-id";
    id.textContent = session.session_id || "unknown";
    row.append(top, id);
    elements.sessionList.append(row);
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
  const response = await fetch(demoUrl("/api/demo/sessions"), {
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
    const response = await fetch(demoUrl("/api/demo/session/items"), { cache: "no-store" });
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
    const response = await fetch(demoUrl("/api/demo/sessions"), { cache: "no-store" });
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
    const response = await fetch(demoUrl("/api/demo/session/items"), {
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

async function invokeSync(payload) {
  const response = await fetch("/api/invocations", {
    method: "POST",
    credentials: "same-origin",
    headers: invocationHeaders(),
    body: JSON.stringify(invocationPayload(payload)),
  });
  const result = await jsonResponse(response);
  const output = agentResult(result);
  addEvent("response", result);
  if (output.session_id) setSessionId(output.session_id);
  handleOutput(output.output);
  return output;
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
  const response = await fetch("/api/invocations", {
    method: "POST",
    credentials: "same-origin",
    headers: invocationHeaders(),
    body: JSON.stringify(invocationPayload(payload, { stream: true })),
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
    const response = await fetch(`/api/invocations/${encodeURIComponent(invocationId)}`, {
      cache: "no-store",
      credentials: "same-origin",
    });
    const result = await jsonResponse(response);
    addEvent("background.poll", result);
    if (result.status === "completed") {
      const output = agentResult(result);
      if (output.session_id) setSessionId(output.session_id);
      handleOutput(output.output);
      return output;
    }
    if (result.status === "failed") throw new Error(result.error || "Background invocation failed");
    setStatus(`Background · ${result.status}`, "busy");
  }
  throw new Error("Background invocation did not finish within three minutes.");
}

async function invokeBackground(payload) {
  const response = await fetch("/api/invocations", {
    method: "POST",
    credentials: "same-origin",
    headers: invocationHeaders(),
    body: JSON.stringify(invocationPayload(payload, { background: true })),
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
    await dispatch({ messages: [{ role: "user", content }] }, mode);
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
    const previousSessionId = ensureSessionId();
    setSessionId(newSessionId());
    resetSessionState();
    addEvent("session.new", { session_id: state.sessionId, previous_session_id: previousSessionId });
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
    const response = await fetch(demoUrl(`/api/demo/sessions/${encodeURIComponent(sessionId)}/open`), {
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

function renderModels(models) {
  const available = models?.available?.length ? models.available : [models?.default].filter(Boolean);
  state.model = available.includes(state.model) ? state.model : models?.default || available[0] || "";
  elements.modelSelect.replaceChildren();
  for (const name of available) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.selected = name === state.model;
    elements.modelSelect.append(option);
  }
  // A single choice is informational, not a decision to make.
  elements.modelSelect.disabled = state.busy || available.length <= 1;
  sizeModelSelect();
}

// Native selects size to their widest option, which leaves a gap between a short
// selected name and the chevron. Size the control to the selected option instead.
function sizeModelSelect() {
  const select = elements.modelSelect;
  const option = select.selectedOptions && select.selectedOptions[0];
  if (!option) return;
  const probe = document.createElement("span");
  const style = getComputedStyle(select);
  probe.style.cssText = "position:absolute;visibility:hidden;white-space:pre;";
  probe.style.fontFamily = style.fontFamily;
  probe.style.fontSize = style.fontSize;
  probe.style.fontWeight = style.fontWeight;
  probe.style.letterSpacing = style.letterSpacing;
  probe.textContent = option.textContent;
  document.body.append(probe);
  const width = probe.getBoundingClientRect().width;
  probe.remove();
  // selected text + left padding (8) + room for the chevron (~26)
  select.style.width = `${Math.ceil(width) + 34}px`;
}

async function loadModels() {
  const response = await fetch(demoUrl("/api/demo/models"), { cache: "no-store" });
  renderModels(await jsonResponse(response));
}

async function loadConfig() {
  try {
    const response = await fetch(demoUrl("/api/ui/config"), { cache: "no-store" });
    const config = await jsonResponse(response);
    state.config = config;
    state.instanceId = config.instance_id;
    setSessionId(config.session_id);
    renderModels(config.models);
    void loadModels().catch((error) => addEvent("models.error", { message: String(error) }));
    setViewer(config.viewer);
    elements.streamingMode.textContent = config.streaming.mode;
    elements.backgroundMode.textContent = config.background.mode;
    elements.sessionMode.textContent = config.session.mode;
    elements.memoryMode.textContent = config.memory.enabled ? `Managed · actor ${config.memory.actor}` : "Not connected";
    setCapability(
      elements.streamingStatus,
      config.streaming.enabled,
      config.streaming.enabled
        ? `Streaming responses over ${config.streaming.transport} use ${config.streaming.persistent ? "the Runtime Store." : "an in-process Runtime Store."}`
        : "Streaming is disabled for this deployment.",
    );
    setCapability(
      elements.backgroundStatus,
      config.background.enabled,
      config.background.enabled
        ? `Background invocations use ${config.background.persistent ? "the Runtime Store." : "an in-process Runtime Store."}`
        : "Background invocations are disabled.",
    );
    setCapability(
      elements.sessionStatus,
      config.session.managed ? true : config.session.history ? "degraded" : false,
      config.session.managed
        ? `Connected to Session Store "${config.session.store}" for actor ${config.session.actor}. State is durable and shareable.`
        : "Session state is kept in process and resets when the app restarts.",
      config.session.managed ? null : "mason sessions bind <store-name>",
    );
    setCapability(
      elements.memoryStatus,
      config.memory.enabled,
      config.memory.enabled
        ? `Connected to ${config.memory.store} for actor ${config.memory.actor}.`
        : "No long-term memory is connected.",
      config.memory.enabled ? null : "mason memory bind <store-name>",
    );
    elements.tracingMode.textContent = config.tracing?.enabled ? "Connected" : "Not configured";
    setCapability(
      elements.tracingStatus,
      config.tracing?.enabled ? true : "degraded",
      config.tracing?.enabled
        ? "Every run is traced to an MLflow experiment."
        : "Tracing is off. Set an MLflow destination and experiment to record traces.",
      config.tracing?.enabled ? null : "mason tracing bind --experiment-name <path>",
      config.tracing?.enabled && config.tracing?.url
        ? { href: config.tracing.url, label: "View traces in MLflow" }
        : null,
    );
    elements.newSession.disabled = state.busy;
    elements.refreshSession.disabled = state.busy || !config.session.history;
    elements.resumeSession.disabled = state.busy || !config.session.durable;
    elements.rejectSession.disabled = state.busy || !config.session.durable;
    elements.sessionStoreLabel.textContent = config.session.managed
      ? `${config.session.store} · actor ${config.session.actor} · the browser session ID keys transcript and checkpoint state.`
      : config.session.history
        ? "Messages load from the in-process LangGraph checkpoint for the current browser session."
        : "Session history is unavailable.";
    addEvent("runtime.config", config);
    void refreshSessionView({ hydrateChat: true });
    // The memory pane loads lazily when opened; refresh it if it is already open.
    if (!elements.memoryPane.hidden) void loadMemory();
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
  updateSendState();
  try {
    await sendText(text);
  } catch {
    elements.promptInput.value = text;
    updateSendState();
  }
});

elements.promptInput.addEventListener("input", () => {
  elements.promptInput.style.height = "auto";
  elements.promptInput.style.height = `${Math.min(elements.promptInput.scrollHeight, 180)}px`;
  updateSendState();
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

elements.modelSelect.addEventListener("change", () => {
  state.model = elements.modelSelect.value;
  sizeModelSelect();
  addEvent("model.selected", { model: state.model });
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
  state.selectedEventId = null;
  renderEvents();
});

// Capabilities expand inline on click; Memory instead opens its own pane.
document.querySelectorAll(".capability").forEach((button) => {
  button.addEventListener("click", () => {
    const item = button.closest(".capability-item");
    if (item?.dataset.capability === "memory") {
      openMemoryPane();
      return;
    }
    const panel = item?.querySelector(".capability-detail");
    const expanded = item.classList.toggle("expanded");
    button.setAttribute("aria-expanded", String(expanded));
    if (panel) panel.hidden = !expanded;
  });
});

// Memory pane controls.
elements.memoryBack.addEventListener("click", closeMemoryPane);
elements.memoryRefresh.addEventListener("click", () => loadMemory());
elements.memoryActor.addEventListener("input", () => {
  window.clearTimeout(memoryState.searchTimer);
  memoryState.searchTimer = window.setTimeout(() => void loadMemory(), 300);
});
elements.memorySearch.addEventListener("input", () => {
  memoryState.query = elements.memorySearch.value;
  window.clearTimeout(memoryState.searchTimer);
  memoryState.searchTimer = window.setTimeout(() => void loadMemory(), 300);
});

// Memory modal: close on ×, backdrop click, or Escape.
elements.memoryModalClose.addEventListener("click", closeMemoryModal);
elements.memoryModal.addEventListener("click", (event) => {
  if (event.target === elements.memoryModal) closeMemoryModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.memoryModal.hidden) closeMemoryModal();
});

// Sessions search.
elements.sessionSearch.addEventListener("input", () => {
  state.sessionFilter = elements.sessionSearch.value;
  renderSessionList();
});

// Events view toggles.
elements.eventViewLogs.addEventListener("click", () => setEventView("logs"));
elements.eventViewList.addEventListener("click", () => setEventView("list"));

// Events filter pills (list view only).
elements.eventFilters.querySelectorAll(".event-filter").forEach((pill) => {
  pill.addEventListener("click", () => {
    state.eventFilter = pill.dataset.filter;
    elements.eventFilters
      .querySelectorAll(".event-filter")
      .forEach((item) => item.classList.toggle("active", item === pill));
    renderEvents();
  });
});

// Copy the selected event payload from the list-view detail pane.
elements.copyEventDetail.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(elements.eventDetailJson.textContent || "");
  } catch {
    /* clipboard may be unavailable */
  }
});

// Header subtitle: framework · host (host is only known client-side).
if (elements.brandSub) elements.brandSub.textContent = `${AGENT_FRAMEWORK} · ${window.location.host}`;
elements.approveAction.addEventListener("click", () => resume("approve"));
elements.rejectAction.addEventListener("click", () => resume("reject"));
elements.resumeSession.addEventListener("click", () => resume("approve"));
elements.rejectSession.addEventListener("click", () => resume("reject"));

loadConfig().catch((error) => {
  appendError(error);
  elements.streamingMode.textContent = "Unavailable";
  elements.backgroundMode.textContent = "Unavailable";
  elements.sessionMode.textContent = "Unavailable";
  elements.memoryMode.textContent = "Unavailable";
  elements.tracingMode.textContent = "Unavailable";
});
