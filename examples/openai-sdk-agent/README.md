# OpenAI SDK Recovery Modes

These three Databricks Apps run the same OpenAI Agents SDK PR-review agent and
the same MLflow `@invoke()` / `@stream()` handlers. Only two
`LongRunningAgentServer` options change:

| App | `auto_recovery` | `sse_replay` | Resume input | Durable event rows |
| --- | --- | --- | --- | --- |
| [`framework_recovery_sse/`](./framework_recovery_sse/) | `True` | `True` | Original request plus prose built from prior stream events; rotated SDK session | Yes |
| [`agent_recovery_sse/`](./agent_recovery_sse/) | `False` | `True` | Fixed recovery prompt; same SDK session restores its transcript | Yes, for client replay only |
| [`agent_recovery_polling/`](./agent_recovery_polling/) | `False` | `False` | Fixed recovery prompt; same SDK session restores its transcript | No |

Heartbeat, stale-attempt claiming, and process-start recovery run in all three
modes. `auto_recovery` controls who restores agent context, not whether a stale
attempt is restarted.

```text
openai-sdk-agent/
├── framework_recovery_sse/   # app.py, app.yaml, README.md
├── agent_recovery_sse/       # app.py, app.yaml, README.md
├── agent_recovery_polling/   # app.py, app.yaml, README.md
└── shared/                   # identical agent, handlers, sessions, lifecycle
```

## Recovery hook

Every claimed stale attempt passes through `LongRunningAgentServer.on_resume`.
The default implementation has two policies:

```python
LongRunningAgentServer(auto_recovery=True)
# Build [RECOVERY] prose from the prior attempt's event log and rotate the
# conversation ID so the handler opens a clean SDK session.

LongRunningAgentServer(auto_recovery=False)
# Keep the original session ID and replace request.input with one fixed
# [RECOVERY] prompt. The agent SDK session store supplies the transcript.
```

Applications can subclass the server and override `on_resume` for another
contract.

## Event and result persistence

The server writes `agent_server.messages` only when at least one consumer needs
the event log:

```text
event_log_enabled = auto_recovery or sse_replay
```

- Framework-managed recovery reads prior events to construct recovery prose.
- SSE replay reads events for `starting_after` cursor recovery.
- When both options are false, the messages table remains available for schema
  compatibility but receives no rows for that response.

Polling does not depend on event rows. Every terminal Responses payload is
stored in `agent_server.responses.response`, alongside status, attempt,
heartbeat, and original request.

The SDK transcript is independent:

| Owner | Schema | Purpose |
| --- | --- | --- |
| `LongRunningAgentServer` | `agent_server.responses` | Request, final response, handler mode, heartbeat, attempt, status |
| `LongRunningAgentServer` | `agent_server.messages` | Optional stream-event log |
| OpenAI Agents SDK | `LAKEBASE_SESSION_SCHEMA.agent_messages` | Agent, model, tool-call, and tool-output transcript |

## Shared agent code

- `shared/review_agent.py` contains the deterministic PR CUJs and shell tool.
- `shared/handlers.py` adapts OpenAI Agents SDK output to MLflow Responses
  events.
- `shared/sessions.py` creates `AsyncDatabricksSession` instances.
- `shared/server_factory.py` contains common Lakebase, lifespan, and dynamic
  App port wiring.
- Each mode folder contains only its `app.py`, `app.yaml`, and a short README.
  The `app.py` files intentionally differ only in the two policy flags.

## HTTP tests

Use background streaming for either replay-enabled app:

```bash
curl -N -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "background": true,
    "stream": true,
    "input": [{"role": "user", "content": "Execute the complete PR CUJ."}],
    "custom_inputs": {
      "session_id": "framework-sse-test",
      "pr_url": "https://github.com/databricks/databricks-ai-bridge/pull/459",
      "minimum_minutes": 0
    }
  }'
```

Reconnect with the response ID and last observed sequence:

```bash
curl -N "$APP_URL/responses/$RESPONSE_ID?stream=true&starting_after=$LAST_SEQUENCE" \
  -H "Authorization: Bearer $APP_TOKEN"
```

Use background polling for the no-replay app:

```bash
curl -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "background": true,
    "input": [{"role": "user", "content": "Execute the complete PR CUJ."}],
    "custom_inputs": {
      "session_id": "agent-polling-test",
      "pr_url": "https://github.com/databricks/databricks-ai-bridge/pull/459",
      "minimum_minutes": 0
    }
  }'

curl "$APP_URL/responses/$RESPONSE_ID" \
  -H "Authorization: Bearer $APP_TOKEN"
```

`stream=true` is rejected for a background request or retrieval when
`sse_replay=False`.

## Deploy

The bundle defines all three Apps. Give each App its own Lakebase branch and
database resource. Every App service principal creates the fixed
`agent_server` schema, so sharing one branch would make the first principal the
schema owner and prevent the other two from initializing it. The SDK session
schemas are also separate.

Each mode folder has its own runtime `app.yaml`. The root bundle includes the
shared implementation and embeds all three runtime configurations, so it is
the easiest way to deploy the comparison together. To deploy one mode through
a direct App upload, copy that mode's `app.yaml` to this directory first so
the uploaded source still includes `shared/`.

```bash
databricks apps deploy -t dev --profile <PROFILE> \
  --var="framework_recovery_sse_lakebase_branch=projects/<project>/branches/<framework-branch>" \
  --var="framework_recovery_sse_lakebase_database=projects/<project>/branches/<framework-branch>/databases/<database>" \
  --var="agent_recovery_sse_lakebase_branch=projects/<project>/branches/<agent-sse-branch>" \
  --var="agent_recovery_sse_lakebase_database=projects/<project>/branches/<agent-sse-branch>/databases/<database>" \
  --var="agent_recovery_polling_lakebase_branch=projects/<project>/branches/<agent-polling-branch>" \
  --var="agent_recovery_polling_lakebase_database=projects/<project>/branches/<agent-polling-branch>/databases/<database>" \
  --var="openai_secret_scope=<scope>" \
  --var="openai_secret_key=<key>"
```

When testing this unreleased branch, package it as a wheel and include that
wheel in the App source instead of resolving `databricks-ai-bridge` from PyPI.
