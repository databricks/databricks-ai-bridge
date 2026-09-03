# Mason OpenAI Chat App Overlay

`mason init --framework openai` copies this framework-specific overlay after the base `agent-openai`
template (it is included by default; `--disable-chat-app` opts out). It is intentionally not a
post-generation mutation command.

## Installed files

- `ui/` contains the zero-build chat client (shared in shape with the LangGraph template's UI).
- `runtime/ui.py` serves the assets and exposes demo APIs for memory and sessions.
- `runtime/main.py` installs the chat routes on the base FastAPI runtime.
- `tests/test_demo_ui.py` verifies the browser-facing routes.

## Behavior

The capability indicators are automatic. Streaming and background reflect the runtime contract;
Session reflects transcript history; Memory requires `AGENT_MEMORY_STORE`. The transport selector is
the only manual capability choice.

The chat header's model picker lists the workspace's ready chat serving endpoints
(`GET /api/demo/config` → `models`, discovered from `serving_endpoints.list()` and filtered to the
`llm/v1/chat` task), with `agent.agent.MODEL` pinned as the default. Each request sends the selected
endpoint as `model` in the invocation body; the agent is rebuilt per turn, so the picker changes the
model for the next turn without a restart. Discovery is best-effort: if listing is unavailable (e.g.
no permission), the picker falls back to just the default. Omitting `model` uses `MODEL`.

A large workspace can expose thousands of endpoints, so the picker is capped (`_MODEL_LIMIT`, 20)
and ranked — default first, then `databricks-*` foundation models, then other custom/external chat
endpoints — so the cap keeps the canonical choices. The list call is unpaginated and occasionally
fails on big workspaces, so it is retried a few times before falling back to the default.

The UI reads local history from the agent's in-process session (`SQLiteSession`) and managed history
from Session Store items. The Databricks Apps `__Host-databricks-app-router` cookie is both the sticky
routing key and the application session id; request bodies do not carry `session_id`. Local
development uses the runtime's `mason-local-session` fallback cookie. TODO: switch to `X-Routing-Key`
when Apps supports it.

The Sessions card calls `POST /api/session/new` to replace that routing cookie with a fresh UUID and
start an empty conversation. With a managed Session Store, `GET /api/demo/sessions` lists the most
recent sessions for the configured actor and each Open action calls
`POST /api/demo/sessions/{session_id}/open`. Opening a listed session verifies that it belongs to the
same actor, replaces the routing cookie, and reloads its transcript. In local in-memory mode only the
current browser session can be listed because there is no shared session index.

Transcript responses include only user, assistant, tool, system, and human-decision message items;
non-message items remain in Session Store but are never returned to the chat UI.

Human-in-the-loop pauses are **in-process only**: a paused run (the Agents SDK `RunState`) is held in
memory by `agent/agent.py`, not in the session transcript, so — unlike the LangGraph template — it is
not durable even with a managed Session Store, and the unmanaged history path never reports pending
interrupts. Resume a pause on the same process that created it.
