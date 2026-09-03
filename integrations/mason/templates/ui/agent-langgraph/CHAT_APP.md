# Mason LangGraph Chat App Overlay

`mason init --framework langgraph` copies this overlay after the base template. Use
`--disable-chat-app` to generate an API-only project.

## Installed files

- `ui/` contains the zero-build chat client.
- `runtime/ui.py` serves the assets and exposes demo APIs for memory and sessions.
- `runtime/main.py` installs the chat routes on the SDK-provided FastAPI application.
- `tests/test_demo_ui.py` verifies the browser-facing routes.

## Behavior

The capability indicators are automatic. Streaming and background reflect the SDK runtime contract;
Session reflects checkpoint history; Memory requires `AGENT_MEMORY_STORE`; and Durability reflects
whether the application is using a persistent runtime store. The transport selector is the only
manual capability choice.

The UI reads local history from the LangGraph checkpoint and managed history from Session Store
items. The browser uses the Databricks Apps routing cookie, or the server's local fallback cookie,
to keep its selected UI session. Programmatic clients must preserve the same cookie; body
`session_id` values are ignored.

The Sessions card calls `POST /api/session/new` to replace that routing cookie with a fresh UUID and
start an empty conversation. With a managed Session Store, `GET /api/demo/sessions` lists the most
recent sessions for the configured actor and each Open action calls
`POST /api/demo/sessions/{session_id}/open`. Opening a listed session verifies that it belongs to the
same actor, replaces the routing cookie, and reloads its transcript and pending LangGraph state. In
local in-memory mode only the current browser session can be listed because there is no shared
session index.

Transcript responses include only user, assistant, tool, system, and human-decision message items;
checkpoint fragments remain in Session Store but are never returned to the chat UI. Crash recovery
uses the application's internal durable runtime and the same registered recovery hook as every
other invocation.
