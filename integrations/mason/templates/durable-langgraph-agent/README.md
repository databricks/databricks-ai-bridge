# Durable LangGraph Agent

A `ChatDatabricks` LangGraph agent hosted by `databricks_mason.DurableAgentApp`. Agent output and
token events are persisted through `DurableAgentContext.emit`, so clients can replay them after a
disconnect or process restart.

## Run locally

```bash
mason dev
```

Invoke it with a client-generated UUID and LangChain message input:

```bash
ROUTING_KEY=$(uuidgen)
INVOCATION_ID=$(uuidgen)

curl -sS http://localhost:8000/api/invocations \
  -H "Cookie: __Host-databricks-app-router=$ROUTING_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":[{\"role\":\"user\",\"content\":\"What time is it? Use your tool.\"}]}"
```

Set `stream: true` to receive persisted LangGraph token and message events as SSE. Set
`background: true` to return immediately and poll `/api/invocations/<id>`. Combining both returns
both `status_url` and `events_url`.

The client owns the invocation `id`. Retrying the same input with the same ID returns the persisted
invocation; reusing the ID with different input returns `409`.

## Test crash recovery

The sample `wait_for_seconds` tool creates a safe window in which to restart a deployed app:

```json
{
  "input": [
    {
      "role": "user",
      "content": "Use wait_for_seconds to wait for 90 seconds, then say recovery test complete."
    }
  ],
  "background": true,
  "stream": true
}
```

After the first attempt becomes active, stop and start the Databricks App. Lakebase keeps the active
execution and emitted events. Once its heartbeat is stale, the new process claims attempt 2 and
calls `@app.on_recovery`. The recovery handler adds a system message explaining that the pod crashed
and asks the agent not to repeat the deliberate wait.

Execution durability does not make external side effects exactly-once. Agent tools must still be
idempotent because a recovered attempt can repeat work performed before the crash.

## Deploy

`mason init --framework langgraph --durability` scaffolds this template. Deploy with an explicit
profile:

```bash
mason --profile <profile> deploy durable-langgraph-agent --source .
```

At deploy time Mason attaches one Lakebase database for runtime tables. It reuses a bound Session
Store database when present; otherwise it reuses or provisions `<app>-durability`. Runtime tables
live in the app-owned `databricks_mason_runtime_<app-hash>` schema. The standard LangGraph and
OpenAI templates use the same durable transport with their full framework/session/UI examples.
