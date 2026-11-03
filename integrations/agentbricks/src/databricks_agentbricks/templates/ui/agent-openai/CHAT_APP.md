# Transcript Chat App Overlay

`agentbricks init --framework openai` and `agentbricks init --framework maf` copy this shared overlay
after the selected agent template. It is included by default; `--disable-chat-app` opts out.
It is intentionally not a post-generation mutation command.

## Installed files

- `ui/` contains the zero-build chat client (shared in shape with the LangGraph template's UI).
- `runtime/ui.py` serves the assets and exposes demo APIs for memory and sessions.
- `runtime/main.py` installs the chat routes on the base FastAPI runtime.
- `tests/test_demo_ui.py` verifies the browser-facing routes.

## Behavior

The capability indicators are automatic. Streaming and background reflect the runtime contract;
Session reflects transcript history; Memory requires `AGENT_MEMORY_STORE`; Traces reflects the MLflow
tracing config (destination + experiment). Selecting Memory opens a pane for browsing the store's
entries (searchable, filterable by actor, with a modal for each entry); the other indicators expand
in place, and Traces links out to the MLflow experiment. The transport selector is the only manual
capability choice.

The composer's model picker lists the workspace's Unity Catalog AI Gateway chat model services — the
`system.ai.*` model services from `GET /api/2.1/unity-catalog/model-services?parent=schemas/system.ai`
(embeddings-only services filtered out), exposed as `GET /api/demo/models` and pinned to
`agent.agent.MODEL` as the default. Each request sends the selected model as `model` in the
invocation body; the agent is rebuilt per turn, so the picker changes the model for the next turn
without a restart. Discovery is best-effort: if listing is unavailable (e.g. `system.ai` isn't
readable), the picker falls back to just the default. Omitting `model` uses `MODEL`.

The agent calls the chosen model through the gateway (`<host>/ai-gateway/mlflow/v1`) rather than
`/serving-endpoints`, so `MODEL` is a `system.ai.*` model service name. The picker includes every listed
model, with the project default first and the rest alphabetical. Selecting a model changes only the
model for this page; the project agent's instructions and tools still run. Reload to reset.

**Foreground** displays the complete answer; **Background** checks the run status until it
finishes, without a browser deadline. Failed or disconnected streams show an error. Chat displays
answer text blocks as Markdown, with readable reasoning in a collapsed **Reasoning** section.
Signatures and encrypted reasoning are excluded from the chat display. Raw HTML is escaped
and images are not loaded; Copy preserves the original Markdown.

The UI reads both local and managed history through `agent.agent.read_history(session_id, actor)`.
The native adapter translates its transcript to the browser's common message shape. Keep this export
when adapting the agent. The browser keeps a stable application session UUID in local storage,
sends it as every durable invocation's top-level `session_id`, and uses the same value in the
independent `X-Routing-Key` header for sticky routing; each turn gets a separate invocation UUID.

The Sessions card creates new session UUIDs in the browser. With a managed Session Store,
`GET /api/demo/sessions` lists the most recent sessions for the signed-in actor and each Open action
calls `POST /api/demo/sessions/{session_id}/open`. Opening a session verifies ownership and reloads
its transcript. In local in-memory mode only the current browser session can be listed because there
is no shared session index.

Transcript responses include only user, assistant, tool, system, and human-decision message items;
non-message items remain in Session Store but are never returned to the chat UI.

The native integrations retain different state:

- **OpenAI Agents SDK:** local transcripts use `SQLiteSession`; managed transcripts contain Responses
  items. Paused `RunState` is separate from the transcript. The managed runtime template checkpoints
  it in a separate Session Store session when a store is bound; native callers without checkpoint
  hooks keep it in process. Transcript reads return no interrupts, so reloading history does not
  reconstruct pending approval controls. See [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md) for resume
  and checkpoint requirements.
- **Microsoft Agent Framework:** a native `HistoryProvider` stores complete Message dictionaries,
  including tool calls and results. Local history is scoped to actor and session within the process;
  a bound Session Store retains it across restarts. The template does not implement approval
  continuation or workflow checkpoints. Approval-requiring tools and request-user tool bindings are
  rejected explicitly, so its browser approval controls have no pending actions.
