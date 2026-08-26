# Durable Runtime Library

This cookbook shows the transport-neutral library option.

`agent.py` contains the complete agent. The developer-owned `server.py` must:

- start and stop `DatabricksDurableRuntime`;
- map the application's request into a stable `run_id`, `session_id`, and JSON payload;
- expose submission and polling endpoints; and
- convert persisted events into the application's streaming protocol.

Run it locally with:

```bash
pip install -r requirements.txt
uvicorn server:app --reload
```

Submit background work:

```bash
curl -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{"run_id":"run-1","session_id":"session-1","payload":{"steps":3}}'
```

Poll `/runs/run-1` or replay events from `/runs/run-1/events?after=0`.
