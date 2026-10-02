# OpenAI Agents Chat App Overlay

`agentbricks init --framework openai` copies this framework-specific overlay after the base `agent-openai`
template (it is included by default; `--disable-chat-app` opts out). It is intentionally not a
post-generation mutation command.

## Installed files

- `ui/` contains the zero-build chat client (shared in shape with the LangGraph template's UI).
- `runtime/ui.py` serves the assets and exposes demo APIs for memory and sessions.
- `runtime/model_services.py` is copied from the installing CLI so model discovery matches this UI,
  including when the generated project installs an older released runtime.
- `runtime/main.py` installs the chat routes on the base FastAPI runtime.
- `tests/test_demo_ui.py` verifies the browser-facing routes.

## Behavior

The capability indicators are automatic. Streaming and background reflect the runtime contract;
Session reflects transcript history; Memory requires `AGENT_MEMORY_STORE`; Traces reflects the MLflow
tracing config (destination + experiment). Selecting Memory opens a pane for browsing the store's
entries (searchable, filterable by actor, with a modal for each entry); the other indicators expand
in place, and Traces links out to the MLflow experiment. The transport selector is the only manual
capability choice.

The header identifies the project agent. All three invocation modes call `agent/agent.py` with
its instructions and tools. **Streaming** shows text as it arrives; **Wait for result** displays the
complete answer; **Background** submits a run and checks its status until it finishes. Background
runs have no browser deadline. **Stop waiting** pauses status checks without cancelling the agent;
**Check result** resumes them. Pausing aborts an outstanding status request, without cancelling the
agent. If the run is no longer available, the UI unlocks the session and warns that its outcome
cannot be recovered; check tool side effects before retrying. Transient errors retain the run for
another status check. The run's status link remains visible. Local runs still end when the
server process stops. A failed run or a disconnected stream is shown as an error rather than Ready.

Assistant Markdown (including code blocks, lists and tables) renders during streaming and history
replay. Raw HTML and external images are disabled. Only text/refusal content blocks enter the answer;
explicit reasoning summaries appear separately, while opaque reasoning/signatures remain out of chat.
The vendored renderer is markdown-it 15.0.2 (MIT); its license is in `ui/vendor/`.

The model picker starts with **Project default**, which omits `input.model` so the agent uses
`agent.agent.MODEL`. Selecting another model temporarily overrides the model on this page only;
it preserves the project agent's instructions and tools. Choose Project default or reload to reset.
To change the persistent default, edit `MODEL` in `agent/agent.py`.

Discovery uses the public schema-scoped Unity Catalog model-services API. It independently lists
`system.ai`, the configured default model's schema, and any additional `catalog.schema` names in
`agent.agent.MODEL_SCHEMAS`. All pages are included, without a display cap. Services that advertise
only embeddings, legacy completions, or Responses are excluded because these templates use chat
completions. Missing capability metadata remains eligible; invocation is the authoritative check.

A schema listing failure is visible and does not hide the default or results from other schemas.
**Use another model service…** accepts a full `catalog.schema.model` name even when listing is
unavailable. Both templates invoke these names through `<host>/ai-gateway/mlflow/v1`; this control
does not select legacy serving endpoints. Discovery permission does not imply EXECUTE permission.
An inaccessible or incompatible model reports an invocation error; switch back to Project default
or choose a service for which you have EXECUTE permission.

The UI reads local history from the agent's in-process session (`SQLiteSession`) and managed history
from Session Store items. It keeps a stable application session UUID in browser local storage,
sends it as every durable invocation's top-level `session_id`, and also uses the same value in the
independent `X-Routing-Key` header for sticky routing; each turn gets a separate invocation UUID.

The Sessions card creates new session UUIDs in the browser. With a managed Session Store,
`GET /api/demo/sessions` lists the most recent sessions for the signed-in actor and each Open action
calls `POST /api/demo/sessions/{session_id}/open`. Opening a session verifies ownership and reloads
its transcript. In local in-memory mode only the current browser session can be listed because there
is no shared session index.

Transcript responses include only user, assistant, tool, system, and human-decision message items;
non-message items remain in Session Store but are never returned to the chat UI.

Human-in-the-loop pauses are **in-process only**: a paused run (the Agents SDK `RunState`) is held in
memory by `agent/agent.py`, not in the session transcript, so — unlike the LangGraph template — it is
not durable even with a managed Session Store, and the unmanaged history path never reports pending
interrupts. Resume a pause on the same process that created it.

## Local state and history

Plain `agentbricks dev` keeps conversation state in-process and memory off. To use existing bound
workspace stores before deployment, run `agentbricks --profile <profile> dev --workspace-stores`.
It verifies read access before startup without provisioning resources. Requests can write to those
stores using your credentials; use development stores. Session history and memory then survive
process restart, but local invocation status, background runs, and replay events do not.

The UI reads the agent's authoritative transcript/checkpoint, including turns submitted through the
API. For request-user-auth projects, state routes use the same identity namespace as invocation
routes, while the public browser session id remains unchanged. App-auth agents retain application
actor partitioning and store-level access. Local development uses one local-developer identity.
