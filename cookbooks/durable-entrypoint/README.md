# Durable Entrypoint

This cookbook shows the AgentCore-style embedded runtime option.

`openai_agent.py` contains a minimal OpenAI Agents SDK loop. It stores SDK
conversation state under `context.session_id` and emits native SDK response
events for durable replay. The complete application remains `agent.py`.

`DatabricksDurableApp` supplies the HTTP server, background submission, status
polling, heartbeat/recovery loop, final result storage, and durable SSE replay.

The developer implements only:

```python
@app.entrypoint
async def agent(payload, context):
    output = await run_openai_agent(
        payload["prompt"], context.session_id, context.emit
    )
    return {"output": output}
```

Install `requirements.txt`, set `OPENAI_API_KEY` and
`LAKEBASE_AUTOSCALING_ENDPOINT`, then run it with `python agent.py`. The
generated API exposes `POST /runs`,
`GET /runs/{run_id}`, and `GET /runs/{run_id}/events?after=N`.
