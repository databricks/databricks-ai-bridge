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
├── app.yaml
├── requirements.txt
├── databricks.yml
├── openai_sdk_agent_shared/
│   ├── app.py
│   ├── agent.py
│   ├── handlers.py
│   └── sessions.py
├── event_log_recovery/
│   └── app.py
└── agent_session_recovery/
    └── app.py
```

Both Apps upload this common source directory. `app.yaml` keeps the command,
secret, and Lakebase resource bindings available after a later App restart.
During a bundle deployment, the target sets `RESUME_STRATEGY`; after a plain App
restart, the shared entry point maps the stable App name to the same strategy and
session schema. Both Apps share the agent, handlers, dependencies, and
Lakebase-backed session implementation. The two small strategy-specific
`app.py` files show the equivalent direct wiring.

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

After EVENT_LOG recovery, the client must replace its stored conversation ID
with `conversation_id` from the durable `response.resumed` event before sending
the next turn. A polling client can retrieve the persisted SSE events after
completion to discover the rotated ID.

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
`custom_inputs.session_id` or `context.conversation_id`. If neither is supplied,
the server logs a warning and uses the generated `response_id` as
`context.conversation_id`.

## Durable HITL

HITL uses two durable Responses rather than holding a worker while waiting for
a person. The proposal run completes with `APPROVAL_REQUIRED`; the approval is
a new background streamed request using the same SDK session.

```bash
curl -N -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "background": true,
    "stream": true,
    "input": [{
      "role": "user",
      "content": "PROPOSAL: publish the release notes"
    }],
    "custom_inputs": {"session_id": "approval-session-1"}
  }'
```

After the human reviews the persisted response, submit approval with the same
session ID:

```bash
curl -N -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "background": true,
    "stream": true,
    "input": [{
      "role": "user",
      "content": "APPROVED: wait for 60 seconds, then publish the release notes"
    }],
    "custom_inputs": {"session_id": "approval-session-1"}
  }'
```

The two Responses, their final results, and their stream events are durable.
The SDK session preserves the proposal for the approval turn. If the App stops
during the approved run, `LongRunningAgentServer` reclaims that Response and
uses the configured recovery strategy. Tools remain at-least-once and must be
idempotent.

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

Select the `event_log` or `agent_session` bundle target. Both targets deploy the
common source tree as a different App, recovery strategy, and SDK-session schema.
Provide a Lakebase branch and the OpenAI API-key secret:

Use a separate empty Lakebase database for each App unless both App service
principals have explicitly been granted access to the shared `agent_server`
schema and tables.

```bash
cd cookbooks/openai-sdk-agent

bundle_vars=(
  --var="lakebase_branch=projects/<project>/branches/<branch>"
  --var="lakebase_database=projects/<project>/branches/<branch>/databases/<database>"
  --var="openai_secret_scope=<scope>"
  --var="openai_secret_key=<key>"
)

databricks bundle validate -t event_log --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle deploy -t event_log --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle run recovery -t event_log --profile <PROFILE> "${bundle_vars[@]}"

databricks bundle validate -t agent_session --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle deploy -t agent_session --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle run recovery -t agent_session --profile <PROFILE> "${bundle_vars[@]}"
```
