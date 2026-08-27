# Databricks Durable Runtime

`DatabricksDurableRuntime` is a transport-neutral durability layer for an
idempotent JSON request/response handler. It stores the request, response, and
heartbeat state in Lakebase and re-executes stale work after a process exits.

It does not own agent session history. An OpenAI Agents SDK session, LangGraph
checkpointer, or other harness-managed store remains an executor concern.

## Contract

The caller provides:

- a stable `execution_id`, used as the idempotency and recovery key;
- a JSON-object request; and
- an async executor that returns a JSON-object response.

The executor receives `DurableExecutionContext`. On recovery, `attempt > 1`
and `is_recovery` is true. The runtime passes the exact persisted request on
every attempt and does not add a recovery message or reconstruct history.

```python
from databricks_ai_bridge.durable_runtime import (
    DatabricksDurableRuntime,
    DurableExecutionContext,
)


async def execute(request: dict, context: DurableExecutionContext) -> dict:
    session = create_agent_session(request["session_id"])
    await context.emit({"type": "progress", "message": "agent started"})
    if context.is_recovery:
        result = await resume_from_session(session)
    else:
        result = await start_from_request(request, session)
    return {"output": result}


runtime = DatabricksDurableRuntime(
    execute,
    autoscaling_endpoint="projects/project/branches/branch/endpoints/primary",
)
```

`execution_id` identifies one durable request, not an SDK session. A response
ID or client idempotency key is normally the right value. A session ID can be
used only when the application intentionally allows one durable request per
session. For multi-turn agents, keep the harness session or conversation ID in
the persisted request and assign each invocation its own `execution_id`.

Call `await runtime.start()` during process startup and `await runtime.stop()`
during shutdown. Shutdown cancels local tasks without marking them failed, so a
different process can claim them after the heartbeat becomes stale.

## Input and Output Wiring

- `submit(execution_id, request)` accepts background work and returns persisted
  state immediately.
- `invoke(execution_id, request)` accepts work and waits for the persisted
  response, including when another process owns the attempt.
- `get(execution_id)` returns status, attempt, heartbeat, request, and response.
- `wait(execution_id)` waits for a previously submitted response.
- `events(execution_id, after_sequence=N)` returns persisted events for replay.

Executors may call `await context.emit(event)` to append an ordered JSON event.
Events are stored before delivery, survive worker replacement, and use the returned
sequence number as a replay cursor. HTTP and SSE adapters decide how those generic
events are presented to clients.

## SDK-hosted agent app

`DatabricksDurableApp` supplies the HTTP adapter when the application does not need to own its
server contract:

```python
from databricks_ai_bridge.durable_app import DatabricksDurableApp

app = DatabricksDurableApp()


@app.entrypoint
async def agent(payload, context):
    await context.emit({"type": "progress", "message": "started"})
    return {"output": await run_agent(payload, context.session_id)}
```

`POST /invocations` passes the JSON body to the entrypoint unchanged. `Idempotency-Key` supplies
the durable execution ID and `X-Routing-Key` supplies the session ID; missing values are generated
and returned as response headers. This keeps runtime metadata out of framework-specific request
models.

With no Lakebase or managed Session Store configuration, the app uses process memory for local
development. Lakebase configuration, or `AGENT_SESSION_STORE` in a Mason deployment, selects the
Lakebase store without changing the entrypoint.

Submitting the same ID and same JSON request is idempotent. If the response is
already complete, `invoke` returns the cached response. Reusing the ID with a
different request raises `DurableRequestConflictError`.

An HTTP adapter remains small and transport-specific:

```python
@app.post("/responses")
async def responses(request: RequestModel):
    execution_id = stable_id_from(request)
    payload = executor_payload(request)
    if request.background:
        state = await runtime.submit(execution_id, payload)
        return status_response(state, status_code=202)
    return await runtime.invoke(execution_id, payload)


@app.get("/responses/{execution_id}")
async def retrieve(execution_id: str):
    state = await runtime.get(execution_id)
    return status_or_response(state)
```

The adapter decides how IDs are supplied, which HTTP status shape to return,
and how JSON is converted to framework-specific request or response models.
`executor_payload` should remove transport-only fields such as `background`,
`stream`, polling cursors, or trace-return flags. Changing only the transport
mode must not create a request conflict for the same durable operation.

For background execution, an adapter may generate the ID and return it in the
initial `202` response. For a blocking request that must survive a lost HTTP
connection, the client must supply a stable ID (for example an
`Idempotency-Key`, session ID, or response ID) so it can retry or retrieve the
same operation. The runtime does not hide this client/server recovery contract.

Recommended HTTP error mappings are:

- different request for an existing ID: `409 Conflict`;
- unknown ID on retrieval: `404 Not Found`;
- failed execution: a terminal failed-status payload; and
- blocking wait timeout: a gateway timeout while execution continues in the
  background and remains retrievable by ID.

## Durability Store

The default schema is `databricks_durable_runtime`. It contains two tables:

```text
executions
  execution_id  TEXT PRIMARY KEY
  status        TEXT
  attempt       INTEGER
  heartbeat_at  TIMESTAMPTZ
  request       JSONB
  response      JSONB

execution_events
  sequence_number  BIGSERIAL PRIMARY KEY
  execution_id     TEXT
  attempt          INTEGER
  event            JSONB
```

Recovery is at-least-once. A process can exit after an external side effect but
before persisting its response, so executor tools must tolerate retries where
needed. A compare-and-swap claim on the stale row prevents multiple recovery
attempts from acquiring the same durability ownership.
