# Mason LangGraph Chat App Overlay

`mason init --framework langgraph --enable-chat-app` copies this framework-specific overlay after
the base `agent-langgraph` template. It is intentionally not a post-generation mutation command.

## Installed files

- `ui/` contains the zero-build chat client.
- `runtime/ui.py` serves the assets and exposes demo APIs for memory, sessions, and recovery.
- `runtime/main.py` installs the chat routes on the base FastAPI runtime.
- `tests/test_demo_ui.py` verifies the browser-facing routes.

The durability backend is part of the base agent template, not this UI overlay:
`agent/mason/durability.py`, `agent/mason/recovery.py`, and
`agent/mason/long_running.py`.

## Behavior

The overlay always includes Start App and Stop App. There are no enable-stop or enable-crash flags.
Durability becomes active when `AGENT_SESSION_STORE` is configured.

The capability indicators are automatic. Streaming and background reflect the runtime contract;
Session reflects checkpoint history; Memory requires `AGENT_MEMORY_STORE`; Durability and Heartbeat
require a managed Session Store. The transport selector is the only manual capability choice.

The UI reads local history from the LangGraph checkpoint and managed history from Session Store
items. The Databricks Apps `__Host-databricks-app-router` cookie is both the sticky routing key and
the application session id; request bodies do not carry `session_id`. Local development uses the
runtime's `mason-local-session` fallback cookie. TODO: switch to `X-Routing-Key` when Apps supports
it.

The recovery flow is intentionally a demo, not a production lease implementation: ownership is
last-writer-wins, the worker persists heartbeats, the browser detects a stale owner, and work resumes
at the first node whose output was not checkpointed.
