# Managed runtime internals

The managed runtime owns the HTTP lifecycle for one agent invocation: accepting an idempotent request,
scheduling work, storing status and events, returning results, and replaying events to a reconnecting
client. The public HTTP resource is an **invocation**; its UUID is also the idempotency key.

```text
Client
  │ POST /api/invocations, GET status/events
  ▼
DurableAgentServer
  HTTP adapter and @app.invoke / @app.recover hooks
  ▼
Runtime
  submit, wait/poll, and event replay
  ├── RuntimeStore
  │     session-aware accept/get/claim/complete/fail/events
  │     ├── InMemoryRuntimeStore
  │     └── LakebaseDurableRuntimeStore
  └── InvocationExecutor
        ├── LocalInvocationExecutor
        └── DurableInvocationExecutor
              ├── Heartbeat
              └── RecoveryScheduler
```

## Lifecycle

`DurableAgentServer` uses `Runtime.from_environment(...)` for managed processes and
`Runtime.from_store(...)` when a caller supplies a store explicitly. The selected Runtime Store
determines the execution mode:

- `InMemoryRuntimeStore` uses `Runtime.local()` and `LocalInvocationExecutor`. It supports the same
  foreground, background, polling, streaming, and replay APIs, but state ends with the process.
- `LakebaseDurableRuntimeStore` implements `DurableRuntimeStore`, so `Runtime.durable()` uses
  `DurableInvocationExecutor`. It heartbeats the active attempt and can schedule recovery after a
  worker stops heartbeating.

`RuntimeStore` deliberately has no heartbeat operation or heartbeat state. Those lease details are
owned by `DurableRuntimeStore` and its Lakebase implementation.

`RuntimeStore.accept(invocation_id, request, session_id=...)` defines idempotency. Existing callers
may omit the session. When supplied, the store assigns a queue position and returns the existing
invocation only when its ID, session, and request all match.

Invocations in one session execute serially. A claim succeeds only for the earliest queued
invocation when that session has no active invocation. Recovery claims the same stale active
invocation and preserves its queue position. Invocation and event reads can target either one
invocation or one session; session state returns the active invocation, then the earliest queued
invocation, or `None`. Each claimed attempt receives the saved session ID in its execution context.

All workers sharing a Runtime Store must support session-aware claims before callers start supplying
session IDs. An older worker does not enforce session order and can claim a later queued invocation.

The durable executor always scans for persisted `QUEUED` invocations, including after process
restart, and starts their first attempt through `@app.invoke`. Reads also schedule queued work as a
safety net when a request moves between replicas. `@app.recover` is optional: registering it enables
a separate scanner to replace stale `ACTIVE` attempts. Recovery is at least once, so agent side
effects must be idempotent.

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
