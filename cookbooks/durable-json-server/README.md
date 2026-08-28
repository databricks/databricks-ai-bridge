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

## Background, streaming, and client impact

| Capability | Developer contract | Client contract |
| --- | --- | --- |
| Background | Return the final JSON output. The server starts the task and stores its status and output. | With `background=true, stream=false`, `POST /invocations` returns `202`; poll `GET /invocations/{id}`. |
| Durable streaming | Convert SDK events to JSON and call `await context.emit(event)`. | With `stream=true`, the POST is SSE. Save each SSE `id`, reconnect through `GET /invocations/{id}/events?after=<id>`, and poll for the authoritative output. |

`background=true, stream=true` starts one durable run and immediately opens its
event stream. Disconnecting the client does not cancel the run.

### OpenAI Agents SDK: before and after

The OpenAI Agents SDK has no remote deployment client. Before this server, the
developer defined an application-specific route and translated
`Runner.run_streamed()` events for that route:

```python
result = Runner.run_streamed(agent, input=message, session=session)
async for sdk_event in result.stream_events():
    await send_using_my_server_protocol(sdk_event)

async with http.stream("POST", "/my-agent", json={"message": "hello"}): ...
```

With the generic server, the SDK loop remains inside `agent`; the client adopts
the server's invocation envelope:

```python
async with http.stream(
    "POST",
    "/invocations",
    json={
        "id": "run-1",
        "session_id": "conversation-1",
        "background": True,
        "stream": True,
        "input": {"message": "hello"},
    },
) as response:
    async for line in response.aiter_lines():
        last_event_id = remember_sse_id(line, last_event_id)

final = (await http.get("/invocations/run-1")).json()
```

### LangGraph SDK: before and after

A native LangGraph deployment uses its own threads and runs protocol:

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

This server is framework-neutral, not LangGraph-protocol-compatible. The client
uses `/invocations`, normally passing `thread_id` as `session_id`; the handler
maps `input` to graph input. Keeping `langgraph_sdk` unchanged would require a
LangGraph protocol adapter.

References: [OpenAI Agents SDK streaming](https://openai.github.io/openai-agents-python/streaming/),
[LangGraph background runs](https://docs.langchain.com/langsmith/runs), and
[LangGraph resumable streaming](https://docs.langchain.com/langsmith/streaming).

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
