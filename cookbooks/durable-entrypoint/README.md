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

## Background, streaming, and client impact

| Capability | Developer contract | Client contract |
| --- | --- | --- |
| Background | Return the application's normal JSON result. The app stores status and the final result. | Keep the existing body; add `Idempotency-Key`, `Databricks-Agent-Session-Id`, and `Databricks-Background: true`. A non-streaming request returns `202`; poll `GET /invocations/{run_id}`. |
| Durable streaming | Convert SDK events to JSON and call `await context.emit(event)`. | Add `Databricks-Stream: true`. Save SSE `id` values and reconnect through `GET /invocations/{run_id}/events?after=<id>`. |

`background=true, stream=true` starts one durable run and immediately opens its
event stream. The run ID comes back in `Databricks-Run-Id`; disconnecting does
not cancel the run.

### OpenAI Agents SDK: before and after

The OpenAI Agents SDK is in-process, so a deployed client previously used
whatever route the developer created around `Runner.run_streamed()`:

```python
payload = {"message": "hello"}
async with http.stream("POST", "/invocations", json=payload): ...
```

The request body and route can remain unchanged. The client only adds runtime
headers, then uses the new polling/replay routes after the POST:

```python
async with http.stream(
    "POST",
    "/invocations",
    json=payload,
    headers={
        "Idempotency-Key": "run-1",
        "Databricks-Agent-Session-Id": "conversation-1",
        "Databricks-Background": "true",
        "Databricks-Stream": "true",
    },
) as response:
    run_id = response.headers["Databricks-Run-Id"]
    async for line in response.aiter_lines():
        last_event_id = remember_sse_id(line, last_event_id)

final = (await http.get(f"/invocations/{run_id}")).json()
```

### LangGraph SDK: before and after

A native LangGraph client uses several framework routes, not one invocation
route:

```python
thread = await langgraph.threads.create()
run = await langgraph.runs.create(
    thread["thread_id"], "agent", input={"messages": messages}
)
result = await langgraph.runs.join(thread["thread_id"], run["run_id"])

async for event in langgraph.runs.stream(
    thread["thread_id"], "agent", input={"messages": messages}
):
    consume(event)
```

Headers preserve the body of a single existing endpoint; they do not make the
LangGraph threads/runs protocol compatible. Without a LangGraph adapter, the
client changes to the `httpx` invocation call above and passes `thread_id` in
`Databricks-Agent-Session-Id`.

References: [OpenAI Agents SDK streaming](https://openai.github.io/openai-agents-python/streaming/),
[LangGraph background runs](https://docs.langchain.com/langsmith/runs), and
[LangGraph resumable streaming](https://docs.langchain.com/langsmith/streaming).

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
