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

## Background, streaming, and client impact

| Capability | Library contract | Developer and client contract |
| --- | --- | --- |
| Background | `runtime.submit()` persists and schedules work; `runtime.get()` returns stored status/result. | Developer chooses how the server returns `202`, exposes the run ID, and implements polling. |
| Durable streaming | `context.emit()` stores ordered JSON events; `runtime.events()` reads them by cursor. | Developer defines the SSE/event format and reconnect route; the client must retain the chosen cursor. |

The library does not mandate any HTTP contract. The cookbook chooses
`POST /runs`, `GET /runs/{run_id}`, and `GET /runs/{run_id}/events?after=N` to
make the missing server work visible.

### OpenAI Agents SDK: before and after

The OpenAI Agents SDK has no remote deployment client. A developer may preserve
an existing client exactly, but must map that route to the runtime and decide
how it expresses background and replay semantics. The cookbook instead chooses:

```python
async with http.stream(
    "POST",
    "/runs",
    json={
        "run_id": "run-1",
        "session_id": "conversation-1",
        "background": True,
        "stream": True,
        "payload": {"message": "hello"},
    },
) as response:
    async for line in response.aiter_lines():
        last_event_id = remember_sse_id(line, last_event_id)

final = (await http.get("/runs/run-1")).json()
```

The server-side agent still calls `Runner.run_streamed()` and forwards each SDK
event to `context.emit()`.

### LangGraph SDK: before and after

The native client starts and observes work through LangGraph's own protocol:

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

The library is flexible enough for a developer to recreate those routes and
keep `langgraph_sdk`, but the library does not provide that integration. With
the cookbook server, the client changes to `/runs` and passes `thread_id` as
`session_id`. This is the option with the least mandated client change and the
most developer-owned protocol work.

References: [OpenAI Agents SDK streaming](https://openai.github.io/openai-agents-python/streaming/),
[LangGraph background runs](https://docs.langchain.com/langsmith/runs), and
[LangGraph resumable streaming](https://docs.langchain.com/langsmith/streaming).

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
