# OpenAI Agents SDK Recovery

These two Databricks Apps run the same small OpenAI Agents SDK agent. Only the
`LongRunningAgentServer` recovery strategy changes.

| App | Recovery context after a crash | Agent session |
| --- | --- | --- |
| [`event_log_recovery/`](./event_log_recovery/) | The server appends recovery prose built from the previous attempt's durable events. | Rotates to a fresh session. Requires `@stream()`. |
| [`agent_session_recovery/`](./agent_session_recovery/) | The server sends a fixed recovery prompt and the SDK reloads its own transcript. | Reuses the same session key. |

Both strategies always persist stream events, support `starting_after` SSE
replay, heartbeat active work, and atomically claim stale attempts.

## Files

```text
openai-sdk-agent/
├── event_log_recovery/
│   ├── app.py
│   ├── app.yaml
│   └── handlers.py
├── agent_session_recovery/
│   ├── app.py
│   ├── app.yaml
│   └── handlers.py
└── shared/
    └── src/openai_sdk_agent_shared/
        ├── agent.py
        └── sessions.py
```

The App folders intentionally duplicate the server and handler wiring so the
developer experience is easy to compare. Only the agent and its Lakebase-backed
OpenAI SDK session implementation are shared.

## Resume Hook

`@on_resume()` is optional and runs only after a worker claims a stale
attempt. The transformed request then goes through the same `@invoke()` or
`@stream()` handler used by the original attempt.

```python
from databricks_ai_bridge.long_running import ResumeContext, on_resume


@on_resume()
async def resume(request, context: ResumeContext):
    resumed = await context.default_resume_request(request)
    return resumed
```

The default transformation depends on `resume_strategy`:

- `ResumeStrategy.EVENT_LOG` appends the immediately previous attempt's
  durable events as recovery prose and changes the session key to
  `<original>::attempt-N`.
- `ResumeStrategy.AGENT_SESSION` keeps the original session key and replaces
  the request input with a recovery prompt. The agent SDK supplies the history.

`ResumeContext.previous_attempt_events` exposes the same previous-attempt
events when an application needs a custom transformation.

## Persistence

The example uses three logically separate stores, even when they share one
Lakebase database:

| Store | Tables | Purpose |
| --- | --- | --- |
| Runtime durability | `agent_server.responses` | Original request, terminal response, status, heartbeat, attempt number, and handler mode. |
| Durable event log | `agent_server.messages` | Ordered SSE events and output items used for replay; also used by event-log recovery. |
| Agent session | `<LAKEBASE_SESSION_SCHEMA>.agent_sessions`, `agent_messages` | OpenAI Agents SDK transcript used by agent-session recovery. |

`response_id` identifies a durable HTTP response. The agent session key
identifies SDK history. For agent-session recovery, a client can provide
`custom_inputs.session_id`, `custom_inputs.thread_id`, or
`context.conversation_id`. If none is supplied, the server logs a warning and
uses the generated `response_id` as `context.conversation_id`.

## Try It

Start a background stream. A long tool call makes crash testing easy:

```bash
curl -N -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "background": true,
    "stream": true,
    "input": [{
      "role": "user",
      "content": "Call wait_for_completion for 60 seconds, then say finished."
    }],
    "custom_inputs": {"session_id": "durable-example-1"}
  }'
```

Reconnect after a network interruption or App restart:

```bash
curl -N "$APP_URL/responses/$RESPONSE_ID?stream=true&starting_after=$LAST_SEQUENCE" \
  -H "Authorization: Bearer $APP_TOKEN"
```

Poll the same response without streaming:

```bash
curl "$APP_URL/responses/$RESPONSE_ID" \
  -H "Authorization: Bearer $APP_TOKEN"
```

## Deploy

Build the unreleased bridge package and the shared example package into each App
source directory:

```bash
bash examples/openai-sdk-agent/prepare_deployment.sh
```

Then provide two dedicated Lakebase branches and the OpenAI API-key secret:

```bash
cd examples/openai-sdk-agent

bundle_vars=(
  --var="event_log_lakebase_branch=projects/<project>/branches/<branch>"
  --var="event_log_lakebase_database=projects/<project>/branches/<branch>/databases/<database>"
  --var="agent_session_lakebase_branch=projects/<project>/branches/<branch>"
  --var="agent_session_lakebase_database=projects/<project>/branches/<branch>/databases/<database>"
  --var="openai_secret_scope=<scope>"
  --var="openai_secret_key=<key>"
)

databricks bundle validate --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle deploy --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle run event_log_recovery --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle run agent_session_recovery --profile <PROFILE> "${bundle_vars[@]}"
```
