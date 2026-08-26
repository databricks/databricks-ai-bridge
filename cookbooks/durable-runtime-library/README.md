# Durable Runtime Library

This cookbook shows the transport-neutral library option.

`openai_agent.py` contains a minimal OpenAI Agents SDK loop. It uses the stable
`session_id` for an `AsyncDatabricksSession` and emits the SDK's native response
events into the durable event log. `agent.py` adapts that loop to
`DatabricksDurableRuntime`.

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

Submit background work:

```bash
curl -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{"run_id":"run-1","session_id":"session-1","payload":{"prompt":"Say hello"}}'
```

Poll `/runs/run-1` or replay events from `/runs/run-1/events?after=0`.
