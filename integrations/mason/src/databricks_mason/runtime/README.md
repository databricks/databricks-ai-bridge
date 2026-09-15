# Mason Runtime internals

Mason Runtime owns the HTTP lifecycle for one agent invocation: accepting an idempotent request,
scheduling work, storing status and events, returning results, and replaying events to a reconnecting
client. The public HTTP resource is an **invocation**; its UUID is also the idempotency key.

```text
Client
  │ POST /api/invocations, GET status/events
  ▼
AgentApp
  HTTP adapter and @app.invoke / @app.recover hooks
  ▼
Runtime
  submit, wait/poll, and event replay
  ├── RuntimeStore
  │     accept/get/claim/complete/fail/events
  │     ├── InMemoryRuntimeStore
  │     └── LakebaseDurableRuntimeStore
  └── InvocationExecutor
        ├── LocalInvocationExecutor
        └── DurableInvocationExecutor
              ├── Heartbeat
              └── RecoveryScheduler
```

## Lifecycle

`AgentApp` constructs `Runtime.from_store(...)`. The selected Runtime Store determines the execution
mode:

- `InMemoryRuntimeStore` uses `Runtime.local()` and `LocalInvocationExecutor`. It supports the same
  foreground, background, polling, streaming, and replay APIs, but state ends with the process.
- `LakebaseDurableRuntimeStore` implements `DurableRuntimeStore`, so `Runtime.durable()` uses
  `DurableInvocationExecutor`. It heartbeats the active attempt and can schedule recovery after a
  worker stops heartbeating.

`RuntimeStore` deliberately has no heartbeat operation or heartbeat state. Those lease details are
owned by `DurableRuntimeStore` and its Lakebase implementation.

`RuntimeStore.accept(invocation_id, request)` defines idempotency. The same ID and request returns
the existing invocation; the same ID and different request raises `InvocationConflictError`.

`@app.recover` is optional. A durable Runtime Store still persists results and events without it;
registering the hook enables the recovery scanner to replace stale attempts. Recovery is at least
once, so agent side effects must be idempotent.

## Code map

| Path | Responsibility |
| --- | --- |
| `app.py` | FastAPI adapter and agent-hook registration |
| `runtime.py` | Runtime factories plus submit, polling, and replay facade |
| `store.py` | Shared Runtime Store contract and in-memory implementation |
| `execution.py` | Common attempt execution and process-local scheduler |
| `types.py` | Invocation records, contexts, errors, and JSON types |
| `durability/store.py` | Durable Store extension for leases and recovery claims |
| `durability/execution.py` | Durable scheduling and heartbeat-wrapped attempts |
| `durability/heartbeat.py` | Lease refresh lifecycle |
| `durability/recovery.py` | Stale-attempt scanning and recovery scheduling |
| `durability/lakebase_runtime_store.py` | Lakebase Runtime Store implementation |
