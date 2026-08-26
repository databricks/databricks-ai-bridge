# Durable Entrypoint

This cookbook shows the AgentCore-style embedded runtime option.

The complete application is `agent.py`. `DatabricksDurableApp` supplies the HTTP
server, background submission, status polling, heartbeat/recovery loop, final
result storage, and durable SSE replay.

The developer implements only:

```python
@app.entrypoint
async def agent(payload, context):
    ...
```

Run it with `python agent.py`. The generated API exposes `POST /runs`,
`GET /runs/{run_id}`, and `GET /runs/{run_id}/events?after=N`.
