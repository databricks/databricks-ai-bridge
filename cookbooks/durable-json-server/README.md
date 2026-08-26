# Generic Durable Server

This cookbook shows the supplied-server option.

`openai_agent.py` contains a minimal OpenAI Agents SDK loop. It stores SDK
conversation state under `context.session_id` and emits native SDK response
events for durable replay. `agent.py` passes that loop to
`DatabricksDurableServer`.

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

The fixed API exposes `POST /invocations`, `GET /invocations/{id}`, and
`GET /invocations/{id}/events?after=N`.
