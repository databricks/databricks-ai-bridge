# Durable Entrypoint

This cookbook shows the decorator-style durable app option with an OpenAI
Agents SDK loop.

```python
@app.entrypoint
async def agent(payload, context): ...

@app.on_resume
async def resume_agent(payload, context): ...
```

The app owns background execution, heartbeat recovery, final-result storage,
and cursor-based SSE replay. The agent owns its `AsyncDatabricksSession`, maps
JSON into the OpenAI SDK, and emits SDK events through `context.emit()`.

## Durable HITL flow

HITL is modeled as two durable runs. The runtime does not keep a worker alive
while waiting for a person.

1. A background streamed proposal completes with `requires_action`.
2. The client reviews the persisted result.
3. The client submits approval as another background streamed run using the
   same `session_id`.

Start the proposal and watch its persisted event stream:

```bash
curl -N -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{
    "run_id": "proposal-1",
    "session_id": "approval-session-1",
    "background": true,
    "stream": true,
    "payload": {"action": "publish the release notes"}
  }'
```

Poll `GET /runs/proposal-1`. Its persisted result contains
`result.status=requires_action`. Then approve it:

```bash
curl -N -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{
    "run_id": "approval-1",
    "session_id": "approval-session-1",
    "background": true,
    "stream": true,
    "payload": {
      "action": "publish the release notes",
      "decision": "approve",
      "wait_seconds": 60
    }
  }'
```

Stop the process while the approved action is waiting. A new process reclaims
the stale run, calls `@app.on_resume` with the original payload and same
`session_id`, and appends events to the existing durable stream. Reconnect with:

```bash
curl -N 'localhost:8000/runs/approval-1/events?after=<last-event-id>'
```

Poll `GET /runs/approval-1` for the authoritative final result. External side
effects remain at-least-once and must be idempotent.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export LAKEBASE_AUTOSCALING_ENDPOINT=projects/.../endpoints/...
python agent.py
```
