# LangGraph Agent

This template is a small LangGraph agent hosted by `DatabricksDurableApp`. The SDK owns the
FastAPI routes, background execution, polling, idempotency, recovery heartbeats, and SSE replay.
The application only defines the agent entrypoint.

```python
app = DatabricksDurableApp()


@app.entrypoint
async def agent(request, context):
    ...
```

## Develop

`mason init --profile` writes the selected profile to `.env`, so the generated project can run
without another configuration step.

```bash
mason init --framework langgraph --profile <profile> my-agent
cd my-agent
mason dev
```

The server listens on `http://localhost:8000`. Run the hermetic tests with:

```bash
uv run pytest
```

## Request contract

`POST /invocations` passes the JSON body to the agent unchanged. Runtime context is carried in
headers instead of wrapping or modifying the agent payload:

- `Idempotency-Key`: one durable invocation ID. Reuse it to safely retry the same request.
- `X-Routing-Key`: the conversation/session ID. Reuse it for multi-turn conversations.

Both headers are optional. The runtime generates missing values and returns them in the response
headers. The response body remains agent-defined.

### Sync

```bash
curl -i -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: turn-1" \
  -H "X-Routing-Key: conversation-1" \
  -d '{"input":[{"role":"user","content":"What time is it? Use the tool."}]}'
```

### Streaming

`stream` is a transport flag removed before the request reaches the entrypoint. The response is SSE
and ends with `data: [DONE]`. Events are persisted before delivery, so the same run can be replayed
from `GET /invocations/<id>/events?after=<cursor>`.

```bash
curl -sN -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: turn-2" \
  -H "X-Routing-Key: conversation-1" \
  -d '{"input":[{"role":"user","content":"Count to three."}],"stream":true}'
```

### Background

`background` is also removed before the request reaches the entrypoint. The first response returns
an invocation ID immediately; poll it until the status is terminal.

```bash
curl -sX POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: turn-3" \
  -H "X-Routing-Key: conversation-1" \
  -d '{"input":[{"role":"user","content":"Do something slow."}],"background":true}'

curl -s http://localhost:8000/invocations/turn-3
```

## State and durability

With no extra configuration, local development uses process memory for both runtime state and the
LangGraph checkpointer. Multi-turn requests work while the process is running.

For deployment, Mason can wire a managed Session Store:

```bash
mason deploy \
  --with-session-store my-agent-sessions \
  --create-stores
```

Setting `AGENT_SESSION_STORE` switches both components to the Session Store's Lakebase project:

- `DatabricksDurableApp` persists requests, results, heartbeats, and replayable events.
- `AsyncCheckpointSaver` persists LangGraph conversation and tool state.

The same agent code runs locally and deployed; only the storage selected by the environment changes.

## Project map

| File | Purpose |
| --- | --- |
| `agent/agent.py` | Model, tools, runtime entrypoint, event serialization |
| `agent/session_store.py` | In-memory or managed Lakebase LangGraph checkpointer |
| `agent/tools/sample_tool.py` | Example LangChain tool |
| `app.yaml` | Databricks Apps start command and optional environment |
| `tests/test_agent.py` | Hermetic template smoke tests |

## Deploy

From the project directory, the deployment name defaults to the directory name and Mason reuses the
profile written by `mason init`:

```bash
mason deploy
```

Inspect it with:

```bash
mason deployments get <name>
mason deployments logs <name>
```
