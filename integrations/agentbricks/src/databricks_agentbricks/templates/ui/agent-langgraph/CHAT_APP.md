# LangGraph Chat App Overlay

`agentbricks init --framework langgraph` copies this framework-specific overlay after the base
`agent-langgraph` template (it is included by default; `--disable-chat-app` opts out). It is
intentionally not a post-generation mutation command.

## Installed files

- `ui/` contains the zero-build chat client.
- `runtime/ui.py` serves the assets and exposes demo APIs for memory and sessions.
- `runtime/main.py` installs the chat routes on the base FastAPI runtime.
- `tests/test_demo_ui.py` verifies the browser-facing routes.

## Behavior

The capability indicators are automatic. Streaming and background reflect the runtime contract;
Session reflects checkpoint history; Memory requires `AGENT_MEMORY_STORE`; Traces reflects the MLflow
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
`/serving-endpoints`, so `MODEL` is a `system.ai.*` model service name. The picker is capped
(`_MODEL_LIMIT`, 20) with the default pinned first and the rest alphabetical, so truncation never
drops the configured default. Transient list failures are retried in
`databricks_agentkit.runtime.model_services` before the fallback applies.

The UI reads local history from the LangGraph checkpoint and managed history from Session Store
items. It keeps a stable application session UUID in browser local storage and includes it inside
every durable invocation's `input`; each turn gets a separate invocation UUID. The router cookie is
independent and may still provide sticky replica routing.

The Sessions card creates new session UUIDs in the browser. With a managed Session Store,
`GET /api/demo/sessions` lists the most recent sessions for the signed-in actor and each Open action
calls `POST /api/demo/sessions/{session_id}/open`. Opening a session verifies ownership and reloads
its transcript and pending LangGraph state. In local in-memory mode only the current browser session
can be listed because there is no shared session index.

Transcript responses include only user, assistant, tool, system, and human-decision message items;
checkpoint fragments remain in Session Store but are never returned to the chat UI.
