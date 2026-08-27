# LangGraph Agent Recovery

These two Databricks Apps run the same small LangGraph agent. Only the
`LongRunningAgentServer` recovery strategy changes.

| App | Recovery source | LangGraph thread after a crash |
| --- | --- | --- |
| [`event_log_recovery/`](./event_log_recovery/) | Durable Responses API events are appended to a recovery prompt. | Rotates to a fresh thread. |
| [`agent_session_recovery/`](./agent_session_recovery/) | LangGraph resumes its durable checkpoint. | Reuses the same thread. |

Both strategies persist Responses API events for SSE replay. The LangGraph
checkpointer is a separate store used only for graph execution state.

## Files

```text
langgraph-agent/
├── app.yaml
├── requirements.txt
├── databricks.yml
├── langgraph_agent_shared/
│   ├── app.py
│   ├── agent.py
│   ├── handlers.py
│   └── runtime.py
├── event_log_recovery/
│   └── app.py
└── agent_session_recovery/
    └── app.py
```

Both Apps upload this common source directory. `app.yaml` keeps the command and
Lakebase resource binding available after a later App restart. During a bundle
deployment, the target sets `RESUME_STRATEGY`; after a plain App restart, the
shared entry point maps the stable App name to the same strategy and checkpoint
schema. Both Apps share the agent, event conversion, handlers, dependencies, and
checkpoint setup. The two small strategy-specific `app.py` files show the
equivalent direct wiring.

## Recovery Behavior

| Strategy | Handler input after recovery | Session behavior |
| --- | --- | --- |
| `ResumeStrategy.EVENT_LOG` | Original request plus recovery prose generated from the previous attempt's durable events. | The server removes `custom_inputs.thread_id` and writes a rotated `context.conversation_id`, so LangGraph starts a fresh thread. |
| `ResumeStrategy.AGENT_SESSION` | The shared `@on_resume()` hook marks the request as a native checkpoint resume. The handler calls LangGraph with `None` when the checkpoint has pending work, otherwise it uses the recovery prompt. | The same thread ID reopens the last durable checkpoint and continues pending graph work. |

After EVENT_LOG recovery, the client must replace its stored conversation ID
with `conversation_id` from the durable `response.resumed` event before sending
the next turn. Reusing the original ID would reopen the abandoned attempt's
LangGraph checkpoint. A polling client can retrieve the persisted SSE events
after completion to discover the rotated ID.

The native-resume marker is specific to this cookbook. It demonstrates how
`@on_resume()` can translate the server's framework-neutral recovery lifecycle
into an SDK-specific resume operation.

The graph runs with `durability="sync"`, so each checkpoint is committed before
the next graph step starts. Stream output is emitted at completed model/tool
message boundaries rather than token boundaries. This keeps the adapter small
while still persisting tool calls and tool outputs for recovery and SSE replay.

## Persistence

| Store | Tables | Purpose |
| --- | --- | --- |
| Runtime durability | `agent_server.responses` | Original request, terminal response, status, heartbeat, attempt number, and handler mode. |
| Durable event log | `agent_server.messages` | Ordered Responses API items used for SSE replay and event-log recovery. |
| LangGraph checkpoints | `<LAKEBASE_CHECKPOINT_SCHEMA>.checkpoints` and related LangGraph tables | Graph state used by agent-session recovery. |

`response_id` identifies one durable HTTP response. The LangGraph thread ID
identifies graph state across attempts. A client can provide
`custom_inputs.thread_id`, `custom_inputs.session_id`, or
`context.conversation_id`. If none is supplied for agent-session recovery, the
server uses the generated `response_id` as `context.conversation_id`.

## Durable HITL

HITL uses two durable Responses. The first run creates a proposal and completes
with `APPROVAL_REQUIRED`; the second run carries the human decision and reuses
the same LangGraph thread.

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
    "custom_inputs": {"thread_id": "approval-thread-1"}
  }'
```

After the human reviews the persisted response, approve it with the same thread:

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
    "custom_inputs": {"thread_id": "approval-thread-1"}
  }'
```

The runtime persists both Responses and their stream events; the LangGraph
checkpointer preserves the proposal between turns. If the App stops during the
approved run, the server reclaims it and applies the selected recovery strategy.
Tools remain at-least-once and must be idempotent.

## Try It

Start a background stream. The wait tool makes crash testing predictable:

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
    "custom_inputs": {"thread_id": "durable-langgraph-1"}
  }'
```

Restart the App while the tool is waiting, then reconnect:

```bash
curl -N "$APP_URL/responses/$RESPONSE_ID?stream=true&starting_after=$LAST_SEQUENCE" \
  -H "Authorization: Bearer $APP_TOKEN"
```

For event-log recovery, the resumed event contains the rotated conversation ID.
For agent-session recovery, LangGraph reopens `durable-langgraph-1` and resumes
the pending checkpoint.

## Deploy

Select the `event_log` or `agent_session` bundle target. Both targets deploy the
same shared source tree as a different App, recovery strategy, and checkpoint
schema. The `model_endpoint` variable defaults to `databricks-gpt-5-2` and can
be overridden.
Use a separate empty Lakebase database for each App unless both App service
principals have explicitly been granted access to the shared `agent_server`
schema and tables.

```bash
cd cookbooks/langgraph-agent

bundle_vars=(
  --var="lakebase_branch=projects/<project>/branches/<branch>"
  --var="lakebase_database=projects/<project>/branches/<branch>/databases/<database>"
)

databricks bundle validate -t event_log --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle deploy -t event_log --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle run recovery -t event_log --profile <PROFILE> "${bundle_vars[@]}"

databricks bundle validate -t agent_session --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle deploy -t agent_session --profile <PROFILE> "${bundle_vars[@]}"
databricks bundle run recovery -t agent_session --profile <PROFILE> "${bundle_vars[@]}"
```
