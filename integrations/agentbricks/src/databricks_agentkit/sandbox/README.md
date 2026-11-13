# Sandbox invocation and streaming prototype

The primary CLI journey is `agentbricks deploy APP --sandbox --source PROJECT`: it
publishes source, provisions the App-owned managed Runtime Store, and deploys the orchestrator.
The workspace must support managed Runtime Store provisioning. Installation happens during
session preparation or first invocation. The hosted Workbench uses this source-install runtime.
`environment create` has been removed; environment inspection and debugging remain available.
The builder/wheelhouse experiment remains available only for existing historical manifests.
See the [change map](../../../docs/sandbox-prototype-changes.md) for scope and remaining cleanup.

AgentKit owns source installation, sandbox lifecycle, invocation, and stream
decoding. Databricks Apps hosts that orchestration code; the agent and its
framework run in the sandbox's installed Python environment. No custom image
is required. The local playground uses the same code under the sandbox creator's
OAuth profile.

The App delegates sandbox lifecycle to `SessionSandboxClient.get_or_create(session)`;
that client owns the mapping, creation coordination, owner validation, resume, status and
stop. The orchestrator owns source installation and worker invocation. Allocation locks
cover allocation only. The shared Runtime serializes invocations within each session;
the sandbox invocation guard also prevents execution from overlapping preparation or stop.
An ambiguous sandbox create is reconciled by its stable label before retry.

### Optional deployment warm pool

`agentbricks deploy NAME --sandbox --source PROJECT --sandbox-warm-pool-size 2`
starts one deployment-wide pool of unused prepared sandboxes. Default 0 disables it;
the prototype supports 0–10. `WarmPool` prepares source, dependencies and a resident
handler under the App's identity, without submitting agent invocations. The configured
maximum counts preparing, ready and retiring spares, across all App replicas. Assigned
sessions are additional compute; this setting is not a total deployment compute cap.

`SessionSandboxClient.get_or_create(session)` atomically claims a compatible spare via
`WarmPoolStore`, preserving its physical worker/state directory. The logical session ID
still accompanies each invocation. A health probe verifies that the same initialized
worker is alive and has never served a turn; probing cannot start or resume a worker.
If no spare is available, normal cold preparation proceeds. Claiming wakes the replenisher.
An assigned sandbox never returns to the pool, including after stop. Resume reuses the
session's disk and initializes a new worker as before.

An advisory lock elects one replenisher. Short transactions enforce the capacity limit
and atomically move a spare's mapping to a user session. A generation stamped by deploy
prevents old App replicas from refilling or claiming after a newer policy takes effect.
Redeploy with 0 to drain unused spares; active sessions remain assigned. Unused workers
are checked every 30 seconds and retired after one hour. Initialization failures back off
for 60 seconds. Interrupted initialization is retired, never replayed; ambiguous sandbox
creation retains its capacity reservation until its outcome can be reconciled. If the App
disappears, platform idle timeout is the fallback for stopping compute. Pool readiness is
independent of App readiness; `/api/warm-pool` and the Sandbox panel report capacity.

This pool serves only the default published revision, using App authentication. Custom
handlers can implement `prepare_session(state_dir)` to initialize more than imports.
That hook must not depend on a future request or user identity; it may run for spares that
are never assigned. Opting in authorizes that initialization and idle compute. Model/tool
latency, request-dependent setup, pool misses and stop/resume still incur their usual cost.

**Pool schema:** two additive tables in `SANDBOX_SCHEMA`: `warm_pool_config` (generation,
revision compatibility key, maximum and backoff) and `warm_pool_slots` (inventory, state,
expiry, worker handle and assignment). `sessions.bootstrap_metrics` stores the physical
`worker_state_key` and pool origin after assignment. No existing columns or constraints
change; the shared Runtime Store `invocations` and `invocation_events` schema is unchanged.

The App uses `Runtime`, `DurableInvocationExecutor` and `LakebaseDurableRuntimeStore`
for invocation acceptance, claims, heartbeat ownership, completion and event replay.
`RunManager` supplies the sandbox execution callback; `ControlStore` projects standard
records into the workbench API. It owns no invocation SQL or separate lifecycle.
The shared store's optional `list_invocations` read API supports workbench history.

**Schema impact:** no columns, constraints, indexes or status values change in the
existing Runtime Store `invocations` or `invocation_events` tables. The default schema
is `databricks_agentkit_runtime`. Agent revision, client session/invocation IDs and input
use the existing request JSON. Progress, lifecycle timing and the attempt-file descriptor
use existing event JSON: `sandbox.stage`, `sandbox.lifecycle` and `sandbox.stream.ready`.
Historical `sandbox.text.delta` events remain readable; new tokens stay on sandbox disk.
Runtime IDs hash deployment/revision/client IDs; the Runtime Store session ID scopes FIFO
to that deployment and revision. Recovery uses the existing `recovery_affinity` column.

Sandbox mapping, source revision/hash, execution principal and preparation timings stay
in the unchanged `sessions` table under the deployment's `SANDBOX_SCHEMA`. Source archives
and revision manifests stay in Workspace. The App verifies archive digests and caches them.

On redeploy, the shared runtime initializes its normal tables if absent. The former
`<SANDBOX_SCHEMA>.invocations` table is left untouched and is no longer read or written;
historical runs are not imported into the new chat history. Existing sandbox/session mappings
remain usable. No automatic migration or destructive cleanup runs.

Failed invocations remain terminal. Retrying a failed message submits a new invocation ID;
a transport reconnect resubmits the same ID and input without repeating execution.
The shared executor detects stale workers and claims a fenced recovery attempt, but the
sandbox callback fails it with an explicit diagnostic instead of replaying arbitrary agent
code. The previous remote operation may still be running; inspect it before retrying.
Automatic sandbox/harness recovery remains future work.

Both managed-server templates (`agentbricks init --framework openai` and
`--framework langgraph`) include `runtime.sandbox_entry:invoke`. These shims call
first-party adapters in `sandbox/adapters/`. They accept `input.message` and an
optional `input.model`, use the generated agent's model/tools configuration,
and return a reply plus framework/timing information.

The Workbench composer lists workspace AI Gateway chat models using the App's
identity. Selecting a model sends `input.model` with the next message; “Agent
default” omits the override. Selection lasts for the page and resets on reload
or a change of agent revision. Reconnecting or retrying an invocation preserves
its original input and model. Discovery runs independently of chat readiness;
if it fails, the current selection remains usable and the model list can be reloaded.

The default execution mode keeps one resident Python worker per session. A
lightweight stdlib client runs through exec-sync and submits the invocation over
a private Unix socket. The worker uses the installed agent interpreter, caches
imports, and runs each handler on the same asyncio loop with a fresh contextvars
context. This runtime wrapper is supplied by the orchestrator; the frozen agent
bundle and its installed SDK do not need to change. First-party adapters configure
the agent once per worker and reuse its refreshable App-auth client. Graph
construction, session handles and MCP connections still happen per invocation.
Changing the adapter requires publishing a new bundle with the updated SDK.

First-party adapters return trace IDs/links without synchronously flushing MLflow on each
turn. MLflow exports in the resident worker's background threads; trace availability is eventual.
They default `MLFLOW_MODEL_CATALOG_URI` to an empty string, using MLflow's bundled pricing
catalog so closing the first model span cannot trigger a remote catalog fetch. Explicit
project/environment settings are preserved; cost estimates can be stale or absent when the
installed catalog lacks a model. Token usage and trace export remain enabled.
The optional `shutdown_session(state_dir)` handler hook drains exports on graceful retirement
or SIGTERM. The worker allows at most 10 seconds for the hook before terminating its process
group. Forced termination can lose pending telemetry. One-shot execution drains before exit;
checkpoint/history writes still complete inside each invocation.

Chat previews exclude reasoning blocks, including signed JSON strings emitted by
databricks-langchain 0.21.0. Existing frozen bundles use a read-only compatibility
projection: run history, polling and SSE may add `display_text` for the UI. Raw
results and streams remain available in run details; model checkpoints retain the
signed blocks needed on subsequent turns. Generic JSON results keep their renderer.

Workers stay resident while the sandbox is running, with no separate idle timeout.
Sandbox autostop owns idle compute reclamation. Workers still recycle after 100
invocations. Each turn cancels and drains its unfinished asyncio tasks; failed cleanup preserves a successful reply
and retires the worker. Agent errors also retire it. A 210-second watchdog kills
the worker process group, including ordinary tool/MCP child processes. A subsequent invocation can
start a replacement and reuse installed files and SQLite history/checkpoints.
Startup and session execution locks prevent competing workers, and runtime
version changes replace an idle worker. A lost reply never triggers automatic
resubmission. Agent stdout/stderr goes to `worker.log` in the session state
directory; text deltas retain their separate per-attempt files. For custom code
that requires a fresh process, set `SANDBOX_WORKER_MODE=oneshot` on the App or
local orchestrator. Worker memory is disposable; sandbox disk loss still needs
a separate recovery design. Background tasks and process-global state in custom
agents need explicit lifecycle management. The task sweep does not isolate delayed
event-loop callbacks or executor threads, and detached or elevated subprocesses
can escape process-group cleanup. Custom code that needs isolation between turns
should use oneshot mode.

| Adapter | Native stream | Conversation state between worker processes |
| --- | --- | --- |
| OpenAI Agents | `Runner.run_streamed` / `response.output_text.delta` | Declared Session Store, otherwise `SQLiteSession` in the session directory |
| LangGraph | `graph.astream(stream_mode=["updates", "messages"])` | Declared Session Store, otherwise `AsyncSqliteSaver` in the session directory |

LangGraph's generated `create_agent_graph` accepts a `checkpointer_override`;
normal App serving continues to use `checkpointer()` by default. Sandbox
adapters select the declared managed store or sandbox-local SQLite. Both adapters resolve configured app-auth
tools in the sandbox and reject request-user bindings. Approval/resume and
arbitrary custom HTTP server integration are not implemented here. Other
callable adapters can accept `(request, state_dir, *, emit)` and emit text.

The immutable source's `agent.toml` supplies session, tracing and tool bindings.
Deployment reconciles a declared Session Store, resolves a tracing experiment, and
grants the target App access without replacing existing experiment permissions.
Adapters initialize the experiment before framework tracing and wrap execution in
an application span with the actual input, output and session ID. The result
contains its trace identity and optional browser link. Export/link failures retain
the successful agent result and trace ID with an unverified export status.
Preparation timing includes framework/project imports, source bindings, startup
configuration, session setup and agent construction (including MCP discovery for
LangGraph). These substeps are reported after completion; they are included in the
aggregate preparation duration, rather than additional setup time.

First-party adapters read installed code files in parallel with normal Python
imports on a new worker. Initially they read native extension files from the
installed package inventory, then save the actual imported Python/native file list in the session directory for Stop/resume.
Read-ahead is bounded by time, file count and bytes; it never imports or executes
code itself. Its deadline is cooperative; an in-flight filesystem operation can
take longer. Warm workers skip it. Set `AGENTBRICKS_IMPORT_PREFETCH=0` in the
sandbox process to disable it. `Import cache setup` measures coordination and
manifest work; concurrent reads overlap the framework/project import durations.
Cold Python initialization and per-turn tool discovery still take time.

If the platform cancels a Stop response after accepting shutdown, the App reads
the same sandbox to confirm its state without replaying the command. Only an
observed stopped/terminated state records a successful shutdown.

```mermaid
sequenceDiagram
    participant C as Browser / CLI
    participant A as App hosting AgentKit
    participant B as SessionSandboxClient
    participant W as Workspace publication
    participant D as Lakebase (sessions, invocations, invocation_events)
    participant S as Sandbox / framework adapter
    C->>A: Submit invocation (session + invocation ID)
    A->>D: Shared Runtime accepts and claims invocation
    A->>B: get_or_create(session)
    B->>D: Read mapping; coordinate allocation
    opt Missing or stopped sandbox
        B->>S: Create or resume session sandbox
        B->>D: Save mapping and owner
    end
    B-->>A: Session sandbox
    opt First invocation in this session
        A->>W: Read verified source archive
        A->>S: Upload source, install and validate
    end
    A->>D: Store attempt descriptor (sandbox ID + file path)
    C->>A: Subscribe / reconnect to run SSE
    A->>S: ExecuteCommand(tail -c +1 -F attempt.jsonl)
    A->>S: Start/reuse session worker; submit through local socket
    loop Model text deltas
        S->>S: Append + flush JSONL delta
        S-->>A: gRPC stdout bytes from tail
        A-->>C: SSE chat snapshot (no SQL on the text path)
    end
    S->>S: Flush end record
    S-->>A: Final result from foreground command
    A->>D: Persist final result and status
    A-->>C: Final result snapshot
    A->>S: Terminate only this subscriber's tail reader
```

The stdlib runner writes to
`<session-state>/streams/<invocation-hash>.<attempt-uuid>.jsonl`. It exclusively
creates the private file, so attempts cannot truncate each other. Each text
record is `{"type":"delta","delta":"...","at":<unix-seconds>}`; terminal records are
`{"type":"end","status":"completed"}` or `"failed"`. Persistent workers
write agent logs to `worker.log`; the client returns the final JSON on stdout.
Oneshot execution keeps agent logs on ordinary stdout/stderr. A terminal `completed` record means
the agent returned and its result serialized; it does **not** acknowledge a
Conversation Store commit. A killed worker can leave an incomplete file.

`FileFollower` uses streaming gRPC `ExecuteCommand` with a bounded GNU
`tail -c +1 -F -s 0.05` process. It decodes complete JSONL records across arbitrary
byte and UTF-8 boundaries. It never polls the live file through exec-sync.
Each SSE subscriber owns an independent reader, so any App replica can serve it.
A browser reconnect reads the active attempt file from byte zero and rebuilds its
text preview. A broken gRPC reader leaves execution running; reconnecting SSE opens
a new reader. Subscriptions never allocate/resume sandboxes or hold session locks.
A terminal invocation is read entirely from Runtime Store, even if its sandbox is gone.

Reader teardown cancels gRPC **and** signals the remote reader: canceling the
RPC alone leaves the sandbox process alive. The PID is checked against the
attempt path before signaling. A remote timeout also bounds an orphaned reader
to approximately 320 seconds if the App dies or cleanup fails. It can count as
sandbox activity until then. Attempt files remain on sandbox disk; retention
and disk quota management are still needed for sustained use.

`streaming.py` contains a small private binding to the preview ExecuteCommand
wire contract. `_command.proto` is its checked-in schema subset; regenerate
`_command_pb2.py` with `grpcio-tools>=1.60,<1.63` from the package root:

```sh
python -m grpc_tools.protoc -I src --python_out=src --pyi_out=src \
  src/databricks_agentkit/sandbox/_command.proto
```

The orchestrator needs the `[sandbox]` extra and workspace access to this preview
API over TLS/gRPC port 443. It uses refreshable WorkspaceClient credentials at
call time. No private SDK wheel or Universe source checkout is shipped in the
App. Buffered exec-sync still performs uploads, installation and the foreground
agent command, with its existing 240-second ceiling. SSH execution also uses
gRPC for following the attempt file.

The shared runtime stores one `sandbox.stream.ready` event per attempt with the
sandbox ID, unique file path and invocation start time. It still stores lifecycle
updates and final results. **New text deltas are not written to `invocation_events`.**
Existing stored text events remain readable for historical runs; table schemas do
not change. Each subscriber checks durable state/events once per second to catch
completion, failure or attempt changes; decoded text wakes SSE independently.

Preview output is bounded to 256 Ki characters per subscriber and coalesced in memory.
Slow clients do not block the worker. Runner timestamps preserve time-to-first-text
when replaying current files; old files without timestamps omit that metric.
The browser retains its display buffer, reconnects SSE after transport errors and
immediately flushes final responses. CLI `--stream` consumes the same SSE endpoint;
ordinary status polling returns durable metadata/results, not new partial text.

This follows MAR's **sandbox file → ExecuteCommand(tail -F) → App → SSE** path.
The wire API remains run-scoped chat snapshots, without MAR StreamItems cursors,
committed Session Store items or commit/discard records. Reconnect progressively
reconstructs a preview rather than emitting MAR's explicit replay-at-EOF snapshot.
Final invocation results are durable; default SQLite conversation history and
uncommitted previews are lost with sandbox disk. Arbitrary code is never automatically
replayed. Per-subscriber tail processes, state polling, file retention, quotas and
browser token cadence still require load testing before production use.

`deploy --sandbox` creates/reuses the App-owned managed Runtime Store through the existing
API. The immutable manifest names the store and owning App; deployment receives the
returned branch/database/username. The App discovers the writable endpoint and authenticates
as its service principal. The publisher opens no SQL connection. The Runtime Store's
existing tables and lifecycle are shared with ordinary durable agent deployments.

Managed publications use a distinct namespace; old data and source files are retained
without copying historical sessions. Existing App resource attachments are preserved.
New source is published through `deploy --sandbox`; manually supplied database coordinates
and standalone environment creation are no longer CLI options. This cleanup preserves managed
publication IDs and namespaces, so redeploys reuse current source revisions and session mappings.

Deployment does not access sandbox compute.
Deployment and App startup bind that publication, with no sandbox probe. First invocation
binds the execution principal in Lakebase, creates the sandbox using the App service principal,
installs the saved bundle and executes the adapter. Explicit session preparation can do
setup before invocation; Workbench New Chat uses this path. Each new session repeats setup in its
own sandbox; later turns in that session reuse it. Status reads and loading the page do
not create compute. Stop targets only the selected session; the next turn resumes its disk
and installed environment. A local first
invocation binds it to the local user's identity; cross-identity invocation/deployment is rejected.
Legacy v2 publications retain their shared sandbox. Legacy v1 prepared manifests keep
their explicit preparation and startup validation behavior. Use `deploy --sandbox` to
use per-session compute; existing session mappings are not migrated.

Deployment builds the generic orchestrator SDK wheel from a temporary source snapshot
with a content-derived local version. Apps may reuse an already installed package when
only the bytes at its wheel path change; the version ensures runtime and UI edits are
installed on redeploy. The local checkout version and agent bundle/lockfile stay intact.

The orchestrator serves a copy of the default Agent Bricks template chat UI, adapted
to this prototype's run/snapshot API. Its default Events tab shows run lifecycle,
reported tool calls, and trace links. LangGraph supplies completed native tool-call
IDs and status after the invocation finishes; these are not live tool events.
A separate Sandbox tab shows setup progress, the current session's sandbox status
and ID, startup/stop durations, stop controls, and links to the immutable saved source
and the current Workspace folder. Capability cards distinguish configured bindings
from observed runs and label managed versus sandbox-local state.
`GET /api/agents/{agent_id}/sandbox?session_id=...`
reads the mapping and current sandbox status without acquiring execution ownership,
starting compute, or installing code. It uses the same agent/session key as invocation
and stop. These routes retain the prototype's single-tenant namespace boundary.

`agentbricks environment ssh ENV --app APP --session-id SESSION` opens a developer
terminal through an authenticated WebSocket to the App, then to the exact session
sandbox. Hosted Apps use gRPC/TLS on port 443 for PTY/input/output because gateway
SSH on port 2222 is not reachable from Apps. The debug transport defaults to the
invocation transport; `SANDBOX_DEBUG_TRANSPORT=ssh` selects Lakebox SSH where reachable.
The server resolves the pinned agent revision and checks the developer's
workspace OAuth identity for `CAN_MANAGE` on the configured App, including inherited
user/group permissions. It rejects browser origins and keeps the App's private key
server-side. `--command '...'` preserves stdout, stderr and the exit code without a PTY.
Interactive mode supports resize and Ctrl-C and starts in the agent's source directory
with its virtual environment active. This is a terminal bridge, without SFTP or port
forwarding. HTTPS terminal output drains to stream EOF, including chunks delivered after
an EndEvent. Resize resolves the terminal PID's PTY using `ps` and runs `stty`;
The daemon creates root-owned PTYs, so this uses the sandbox's passwordless `sudo`
for `stty`; direct `/proc/<pid>/fd` access is denied after it drops UID. Disconnect invokes
`TerminateCommand` to reap the command tree. Terminals have a one-hour deadline.
Missing/uninstalled sessions are refused; stopped compute can resume.
Only connection setup holds the session lock. Debugging creates no sandbox, build or
agent invocation. Local debugging must explicitly enable `SANDBOX_ALLOW_LOCAL_DEBUG=true`
and is limited to loopback clients. The `sandbox` extra includes the WebSocket transport.


## Explicit session preparation

`POST /api/agents/{agent_id}/sessions` accepts `{"session_id":"chat-1"}` (or generates
an ID) and returns HTTP 202 with `queued`, `preparing`, `ready`, `failed`, `interrupted`,
or `cancelled` status. Supply a stable ID for retry safety. `GET` on
`/api/agents/{agent_id}/sessions/{session_id}` returns progress, lifecycle timings and
worker identity; it never creates compute. Repeating POST returns the same preparation.
Use `{"session_id":"chat-1","retry":true}` to explicitly retry failed/interrupted work
or prepare a stopped session again. New Chat uses this flow, with Send blocked during
preparation. A visible chat spinner remains through installation and handler initialization,
including resumed sessions. A running sandbox alone does not enable Send: the preparation
must report ready. Stopped sessions expose Resume session, using the same preparation API.
API clients may still invoke directly to install lazily.

Preparation installs the pinned source and starts the persistent worker, then imports the
handler and calls its optional `prepare_session(state_dir)` function attribute. That hook
may be synchronous or asynchronous and runs once per resident worker. The first-party
OpenAI/LangGraph hooks import frameworks and configure source bindings/tracing without
calling a model or executing tools. They report `initialized`; arbitrary handlers without
a hook report `handler-imported`. Per-turn agent construction and MCP discovery remain
on the invocation path. Preparation is not a chat turn and writes no invocation record.

Preparation status lives in the existing session's `bootstrap_metrics.preparation` JSON.
Two bounded scheduler threads pick up queued requests; a separate advisory lock prevents
replicas from preparing the same session. Installation and execution still use the existing
session invocation lease. Queued requests survive App restart. A lost preparation worker
becomes `interrupted` and requires explicit retry; custom hooks are not silently replayed.
Stop serializes against preparation and cancels queued work. Invoke while preparation is
pending returns 409; retry once ready. Readiness is a completed preparation result, not
a guarantee against later sandbox shutdown or worker failure.

`agentbricks deploy APP --sandbox --sandbox-idle-timeout 1800 ...` sets
`SANDBOX_IDLE_TIMEOUT_SECONDS` in the App (default 900; Lakebox range 60–86400).
The policy is applied on create and the first prepare/invoke of an existing sandbox in
each App process. It does not alter source identity, stop an active run at a fixed deadline,
or add a worker idle timer. Debug connections and Lakebox activity affect actual idle time.
