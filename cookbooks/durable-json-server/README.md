# Generic Durable Server

This cookbook shows the supplied-server option.

`agent.py` contains the agent and passes it to `DatabricksDurableServer`. The
server owns the fixed JSON protocol, background submission, blocking invocation,
status polling, heartbeat/recovery loop, final result storage, and durable SSE
replay.

Run it with:

```bash
uvicorn agent:app --reload
```

The fixed API exposes `POST /invocations`, `GET /invocations/{id}`, and
`GET /invocations/{id}/events?after=N`.
