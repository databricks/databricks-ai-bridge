# Durable Header Entrypoint

This cookbook shows the decorator-style app with the application's JSON body
left unchanged. Durable submission metadata is supplied through headers.

```python
@app.entrypoint
async def agent(payload, context): ...

@app.on_resume
async def resume_agent(payload, context): ...
```

The server validates the headers, persists their normalized values with the
original body, and constructs `context` for each attempt. Recovery does not
replay HTTP headers; it rebuilds `context` from the durable run record.

## Durable HITL flow

HITL is modeled as two durable runs. The runtime does not keep a worker alive
while waiting for a person.

1. A background streamed proposal completes with `requires_action`.
2. The client reviews the persisted result.
3. The client submits approval as another background streamed run using the
   same `session_id`.

Start the proposal and watch its persisted event stream:

```bash
curl -N -X POST localhost:8000/invocations \
  -H 'content-type: application/json' \
  -H 'idempotency-key: proposal-1' \
  -H 'databricks-agent-session-id: approval-session-1' \
  -H 'databricks-background: true' \
  -H 'databricks-stream: true' \
  -d '{
    "action": "publish the release notes"
  }'
```

The request body is the application payload, not a runtime envelope. Poll
`GET /invocations/proposal-1`. Its persisted result contains
`result.status=requires_action`. Then approve it:

```bash
curl -N -X POST localhost:8000/invocations \
  -H 'content-type: application/json' \
  -H 'idempotency-key: approval-1' \
  -H 'databricks-agent-session-id: approval-session-1' \
  -H 'databricks-background: true' \
  -H 'databricks-stream: true' \
  -d '{
    "action": "publish the release notes",
    "decision": "approve",
    "wait_seconds": 60
  }'
```

Stop the process while the approved action is waiting. A new process reclaims
the stale run, calls `@app.on_resume` with the original payload and same
`session_id`, and appends events to the existing durable stream. Reconnect with:

```bash
curl -N 'localhost:8000/invocations/approval-1/events?after=<last-event-id>'
```

Poll `GET /invocations/approval-1` for the authoritative final result. External side
effects remain at-least-once and must be idempotent.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export LAKEBASE_AUTOSCALING_ENDPOINT=projects/.../endpoints/...
python agent.py
```
