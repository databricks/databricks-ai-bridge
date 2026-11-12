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
cover allocation only; the separate invocation guard implements the prototype's sequential
turn policy. Two racing allocations return the same sandbox, while overlapping invocations
are rejected as busy. An ambiguous create is reconciled by its stable label before retry.

Lakebase creates only two tables: `sessions` (sandbox ID, source revision/hash, execution
principal, status and bootstrap timings) and `invocations` (request identity/input,
status/result/error, attempt, timing and current text snapshot). Named environment APIs
resolve immutable Workspace manifests, without environment, alias or owner tables.
Source archives and revision manifests live in the Workspace catalog; the App verifies
archive digests and keeps a bounded cache so warm turns avoid source downloads.

The new publication format records `storage_version=2` and `catalog_path`. Republish with
`deploy --sandbox` to adopt it. New deployments use a separate namespace from the prior
eight-table prototype. Existing state is preserved; migration is not automatic, and older
Lakebase-backed publications require their previous runtime or an explicit migration.

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
    participant D as Lakebase (sessions, invocations)
    participant S as Sandbox / framework adapter
    C->>A: Submit invocation (session + invocation ID)
    A->>D: Persist and claim invocation for pinned revision
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
    A->>S: ExecuteCommand(tail -c +1 -F attempt.jsonl)
    A->>S: Start/reuse session worker; submit through local socket
    loop Model text deltas
        S->>S: Append + flush JSONL delta
        S-->>A: gRPC stdout bytes from tail
        A->>D: Flush text snapshot (50 ms timer) + NOTIFY
        D-->>A: Wake SSE reader; load committed snapshot
        A-->>C: SSE snapshot / CLI status response
    end
    S->>S: Flush end record
    S-->>A: Final result from foreground command
    A->>S: Terminate only the tail reader
    A->>D: Persist final result and status
```

The stdlib runner writes to
`<session-state>/streams/<invocation-hash>.<attempt-uuid>.jsonl`. It exclusively
creates the private file, so attempts cannot truncate each other. Each text
record is `{"type":"delta","delta":"..."}`; terminal records are
`{"type":"end","status":"completed"}` or `"failed"`. Persistent workers
write agent logs to `worker.log`; the client returns the final JSON on stdout.
Oneshot execution keeps agent logs on ordinary stdout/stderr. A terminal `completed` record means
the agent returned and its result serialized; it does **not** acknowledge a
Conversation Store commit. A killed worker can leave an incomplete file.

`FileFollower` uses streaming gRPC `ExecuteCommand` with a bounded GNU
`tail -c +1 -F -s 0.05` process. It decodes complete JSONL records across arbitrary
byte and UTF-8 boundaries. It never polls the live file through exec-sync.
If the stream disconnects, the agent continues; after the foreground command
finishes, one catch-up read from the last received byte offset supplies remaining
records. The current implementation does not reconnect the live gRPC stream.

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

An App-side writer coalesces text into a bounded 256 Ki-character snapshot and
flushes pending output on a 50 ms timer, even when no further token arrives.
Database writes do not block the tail reader. Each committed update sends a
PostgreSQL notification with no text payload; SSE subscribers use `LISTEN` to
wake and read the durable snapshot. Subscribing before the initial read avoids
a race with new writes. Reconnects read the latest snapshot; a quiet connection
resynchronizes every 5 seconds. Preview publication failures are logged and the
final invocation result remains authoritative. The browser distributes received text batches
across animation frames with an 80 ms maximum display buffer; final snapshots flush immediately.
Reduced-motion preferences bypass animation. Upstream model/tool pauses remain visible.

This adopts MAR's **file + ExecuteCommand(tail -F)** transport. Remaining
differences: Lakebase still stores bounded partial-text snapshots; browser SSE
uses notification-driven snapshot reads, and CLI `--stream` polls status.
Each SSE subscriber holds a database connection and a blocking read thread;
each streaming invocation writes at most about 20 snapshots/second. Connection
budgets, shared subscription fanout and write amplification need load testing
before scaling. There is no session-level
StreamItems cursor, Conversation Store message/commit/discard protocol, or
automatic harness recovery. Each session in a new v3 publication has its own
sandbox, installed environment, state directory, and execution lock. Different
sessions can run concurrently; overlapping turns in the same session fail busy. SQLite
history/checkpoints survive worker processes, not loss of sandbox disk.

`deploy --sandbox` publishes immutable source and metadata internally before deploying the App.
It creates/reuses the App-owned managed Runtime Store through the existing
Runtime Store API. The immutable manifest names the store and owning App; deployment
configuration receives the returned branch/database/username. The App discovers the
writable endpoint, authenticates as its service principal and initializes exactly two
tables, `sessions` and `invocations`. The publisher opens no SQL connection. This reuses
managed provisioning, not the SDK's invocation/event schema. The 50 ms snapshot writer
and LISTEN/NOTIFY path are unchanged.

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
