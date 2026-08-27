# Durable Runtime Library

This cookbook shows the transport-neutral library option.

`openai_agent.py` contains an OpenAI Agents SDK approval flow. It uses the
stable `session_id` for an `AsyncDatabricksSession` and emits SDK events into
the durable event log. `agent.py` decides how recovered attempts should invoke
the SDK by checking `context.is_recovery`.

The developer-owned `server.py` must:

- start and stop `DatabricksDurableRuntime`;
- map the application's request into a stable `run_id`, `session_id`, and JSON payload;
- expose submission and polling endpoints; and
- convert persisted events into the application's streaming protocol.

Run it locally with:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export LAKEBASE_AUTOSCALING_ENDPOINT=projects/.../endpoints/...
uvicorn server:app --reload
```

## Durable HITL flow

HITL is two durable runs. The first run returns `requires_action`; the second
run carries the human decision and reuses the same SDK session.

Start the proposal as a background stream:

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

Poll `/runs/proposal-1`, review the persisted result, and submit approval:

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

Stop the process during the wait. On restart, the library reclaims the run and
calls the same developer-owned executor with `context.is_recovery=True`. The
developer is responsible for reconnecting the SDK session and choosing the
recovery instruction. Replay from
`/runs/approval-1/events?after=<last-event-id>` and poll `/runs/approval-1` for
the final result.

This option makes the ownership difference explicit: the library stores runs
and events, while the developer implements all HTTP and recovery wiring.
