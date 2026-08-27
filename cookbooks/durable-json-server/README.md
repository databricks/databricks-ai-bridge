# Generic Durable Server

This cookbook shows the explicit generic server option.

The developer provides normal and crash-recovery callbacks:

```python
server = DatabricksDurableServer(agent, on_resume=resume_agent)
app = server.app
```

The server owns the fixed JSON protocol, background submission, blocking
invocation, status polling, heartbeat/recovery loop, final result storage, and
durable SSE replay.

Run it with:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export LAKEBASE_AUTOSCALING_ENDPOINT=projects/.../endpoints/...
uvicorn agent:app --reload
```

## Durable HITL flow

The proposal and approval are separate durable runs. Both use background
execution and persisted streaming; both share one SDK session.

```bash
curl -N -X POST localhost:8000/invocations \
  -H 'content-type: application/json' \
  -d '{
    "id": "proposal-1",
    "session_id": "approval-session-1",
    "background": true,
    "stream": true,
    "input": {"action": "publish the release notes"}
  }'
```

Poll `GET /invocations/proposal-1`. After reviewing its persisted
`requires_action` result, submit the approval:

```bash
curl -N -X POST localhost:8000/invocations \
  -H 'content-type: application/json' \
  -d '{
    "id": "approval-1",
    "session_id": "approval-session-1",
    "background": true,
    "stream": true,
    "input": {
      "action": "publish the release notes",
      "decision": "approve",
      "wait_seconds": 60
    }
  }'
```

Stop the process during the wait. The server reclaims the run and invokes
`resume_agent` with the original input and same session. Reconnect to
`GET /invocations/approval-1/events?after=<last-event-id>` and poll
`GET /invocations/approval-1` for the authoritative result.

External side effects remain at-least-once and must be idempotent.
