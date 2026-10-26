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
may omit the session. When supplied, the store assigns an immutable `session_sequence_number` and
returns the existing invocation only when its ID, session, and request all match.

Invocations in one session execute serially. A claim succeeds only for the earliest queued
invocation when that session has no active invocation. Recovery claims the same stale active
invocation and preserves its sequence number. This is acceptance order, not the event replay cursor.
Invocation and event reads can target either one invocation or one session; session state returns
the active invocation, then the earliest queued
invocation, or `None`. Each claimed attempt receives the saved session ID in its execution context.

The HTTP server accepts an optional top-level `session_id` and passes it to Runtime in every
execution mode. If it is omitted, the invocation remains sessionless. Request-user sessions are
namespaced once before they enter Runtime. The session saved in the Runtime Store and propagated
through the attempt context is the handler's source of truth; neither opaque input nor
`X-Routing-Key` selects Runtime session identity.

All workers sharing a Runtime Store must support session-aware claims before callers start supplying
session IDs. An older worker does not enforce session order and can claim a later queued invocation.

### Worker affinity and request auth

Each Lakebase store generates a fresh worker UUID. Admission atomically saves it with an optional
`recovery_affinity` and database heartbeat while leaving the invocation `QUEUED`. Idempotent retries
preserve this reservation. Normal SQL claims require the owner (or an unassigned legacy row), exact
nullable affinity equality, and session FIFO eligibility; queue scans and reads cannot bypass it.

With `@app.recover`, stale `ACTIVE` attempts and abandoned owned `QUEUED` heads can transfer to another
worker in the same affinity group. Recovery increments the attempt fence and starts at attempt 2 even
if execution never began. Completion/failure refreshes the next queue head's heartbeat so its owner
gets a full grace period after waiting. Without a recovery hook, abandoned reservations remain queued.
Recovery is at least once; agent side effects must be idempotent.

`DurableAgentServer` retains credentials in `_request_auth` while queued, then passes them to the first
attempt and closes them when it ends. Foreground disconnects do not discard queued auth. A lifespan
task releases pending credentials after terminal state or takeover, checking only admitted requests.
Retries on other replicas cannot replace the owner's credentials. Credentials are never persisted.

OBO recovery fails before agent code with `MCP_USER_AUTH_RECOVERY_UNSUPPORTED`, including takeover
before the first attempt. Polling/event replay does not restore auth. New invocations get fresh auth,
but generated LangGraph/OpenAI adapters still reject OBO approval/resume with
`MCP_USER_AUTH_HITL_UNSUPPORTED`; checkpoint resume supports app-auth runs.

Initialization adds nullable `worker_id UUID` and `recovery_affinity TEXT` columns. Legacy rows stay
unassigned in the default group. Upgrade all replicas before relying on affinity: old SDKs issue SQL
without these claim predicates. Named affinity groups never match the default (`None`) group.

Follow-up: a lost response after a committed claim can leave an active row awaiting stale recovery.
Same-worker reattachment for this pre-existing failure mode is not implemented here.

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
