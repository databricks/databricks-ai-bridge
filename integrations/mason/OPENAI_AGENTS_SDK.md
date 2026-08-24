# CUJ: an OpenAI Agents SDK agent on Databricks agent memory + sessions

> **Experimental.** This lives under `experimental/` and has not been through a formal
> review process. The `agents/v1` APIs it wires to are in preview and gated per workspace.

This is the end-to-end critical user journey for taking a **stock OpenAI Agents SDK**
(`openai-agents`) agent and giving it durable **sessions** (per-conversation history) and
**long-term memory** (facts that survive across conversations), both persisted through the
Databricks `agents/v1` APIs (conversation-store) and provisioned/inspected with `mason`.

Verified end-to-end on `eng-ml-inference.staging` (2026-08-18): a two-session run where the
agent is told facts in session 1 (recalled from session history), then answers correctly in a
brand-new session 2 with no shared history — only possible by reading long-term memory back
through the API. The session store persisted the full SDK transcript (user turns, tool calls,
tool outputs, assistant turns); the memory store held the `remember`d facts, retrievable by
`search`.

## What you change vs. a "native" OpenAI Agents SDK agent

**Almost nothing structural.** The `Agent`, its `@function_tool`s, and the `Runner.run(...)`
loop are used verbatim. You touch three integration seams the SDK already exposes, plus one
conditional shim that is really a Databricks-serving bug rather than an SDK requirement.

A vanilla agent:

```python
from agents import Agent, Runner, function_tool, SQLiteSession
agent   = Agent(name="A", instructions="...", tools=[...])   # default model = OpenAI (needs OPENAI_API_KEY)
session = SQLiteSession("conv-1", "conv.db")                  # built-in local session backend
result  = await Runner.run(agent, "hi", session=session)
```

The Databricks version, change-by-change:

| # | What | Native | On Databricks | Is it a "modification"? |
|---|------|--------|---------------|--------------------------|
| 1 | **Model client** | Default OpenAI client / `OPENAI_API_KEY`, Responses API | `OpenAIChatCompletionsModel(model="databricks-…", openai_client=AsyncOpenAI(base_url=f"{host}/serving-endpoints", api_key=<bearer>))` | No — supported config; Databricks serving is OpenAI **chat-completions**-compatible |
| 2 | **Sessions** | `SQLiteSession` / `OpenAIConversationsSession` | Custom `SessionABC` backed by our session-store **items** API (`get_items`↔list, `add_items`↔append) | No — `Session` is a documented extension point |
| 3 | **Long-term memory** | (not a native concept) | Two `@function_tool`s — `remember`→create entry, `recall`→search — over the memory API | No — uses the native tool mechanism |
| 4 | **`strict` tool field** | SDK always emits `function.strict` | **Claude models only:** strip it (Databricks' Anthropic serving path rejects the field). **OpenAI models (`databricks-gpt-*`) need no change.** | Yes — the one genuine override, and it's conditional |
| 5 | **Tracing** | Exports to platform.openai.com | `set_tracing_disabled(True)` (default), or route to MLflow/UC via `mason tracing` | Config — keeps trace data on Databricks |

**Bottom line:** items 1–3 are configuration + supported plug-in seams — the agent is not
modified. Item 4 is the only real behavior override, needed **only for Claude models**; run the
agent on a Databricks-served **OpenAI** model (e.g. `databricks-gpt-5`) and even that disappears.

### Tracing to Unity Catalog (instead of disabling)

The SDK's default tracing exports spans to platform.openai.com. Rather than disable it, route
traces to MLflow / Unity Catalog with the `mason tracing` capability:

```bash
mason tracing setup --catalog my_cat --schema my_schema      # link a UC schema to an experiment (once)
mason tracing instrument --destination my_cat.my_schema      # prints the snippet to paste into the agent
mason deploy my-agent --source ./app --with-traces my_cat.my_schema   # wire a deployed agent
```

`instrument` emits the drop-in replacement for `set_tracing_disabled(True)`:
`mlflow.set_tracking_uri("databricks")` + `mlflow.tracing.set_destination(UCSchemaLocation(...))`
+ `mlflow.openai.autolog()`. This keeps all trace data inside your workspace (no OpenAI egress).
Requires `mlflow[databricks]>=3.9.0` and the "OpenTelemetry on Databricks" preview. Note this is
*observability*, orthogonal to the session/memory stores (agent runtime state).

### The seams, in code

```python
from agents import (Agent, Runner, SessionABC, OpenAIChatCompletionsModel,
                    function_tool, set_tracing_disabled)
from agents.models.chatcmpl_converter import Converter
from openai import AsyncOpenAI
from databricks.sdk import WorkspaceClient
from databricks_mason.client import AgentApiClient  # reuse Mason's REST client as the API layer
from databricks_mason.render import field

set_tracing_disabled(True)
api = AgentApiClient("ml_inference_staging")      # agents/v1 memory + sessions

# (4) Claude-only shim: Databricks' Anthropic serving path 400s on the `strict` key.
#     Omit this block entirely if MODEL is an OpenAI (databricks-gpt-*) endpoint.
Converter.tool_to_openai = classmethod(lambda cls, t: {"type": "function", "function": {
    "name": t.name, "description": t.description or "", "parameters": t.params_json_schema}})

# (2) Sessions: SDK Session protocol -> session-store items API.
class DatabricksSessionStore(SessionABC):
    def __init__(self, store, session_id): self.store, self.session_id = store, session_id
    async def get_items(self, limit=None):
        data = api.list_session_items(self.store, self.session_id, order_by="create_time asc")
        return [field(it, "data") for it in field(data, "session_items") or []]
    async def add_items(self, items):
        if items: api.append_session_items(self.store, self.session_id, list(items))
    async def pop_item(self):     return None   # server PopSessionItem is UNIMPLEMENTED
    async def clear_session(self): pass          # server ClearSessionItems is UNIMPLEMENTED

# (3) Long-term memory as tools.
MEM_STORE, ACTOR = "<memory-store-id>", "ankit"

@function_tool
def remember(fact: str, topic: str) -> str:
    """Persist a durable fact about the user in long-term memory."""
    api.create_memory_entry(MEM_STORE, actor_id=ACTOR, path=f"/{topic}/{fact[:8]}.md", content=fact)
    return "stored"

@function_tool
def recall(query: str) -> str:
    """Search the user's long-term memory for facts relevant to `query`."""
    data = api.search_memory_entries(MEM_STORE, actor_id=ACTOR, query=query, limit=5)
    entries = field(data, "managed_memory_entries") or []
    return "\n".join(f"- {field(e,'content')}" for e in entries) or "No relevant memories."

# (1) Model client -> Databricks Model Serving.
w = WorkspaceClient(profile="ml_inference_staging")
bearer = w.config.authenticate()["Authorization"].split(" ", 1)[1]
model_client = AsyncOpenAI(base_url=f"{w.config.host}/serving-endpoints", api_key=bearer)

agent = Agent(
    name="Memory Agent",
    instructions="Call `remember` for lasting facts; call `recall` before answering about the user.",
    model=OpenAIChatCompletionsModel(model="databricks-claude-sonnet-5", openai_client=model_client),
    tools=[remember, recall],
)

sid = field(api.create_session("<session-store>", ACTOR), "session_id")
result = await Runner.run(agent, "I'm Ankit; I prefer terse answers.",
                          session=DatabricksSessionStore("<session-store>", sid))
```

## API mapping

| SDK / agent concept | Mason client method | agents/v1 REST |
|---|---|---|
| `Session.get_items()` | `list_session_items(store, sid)` | `GET /api/agents/v1/session-stores/{store}/sessions/{sid}/items` |
| `Session.add_items()` | `append_session_items(store, sid, items)` | `POST …/sessions/{sid}/items:append` |
| `remember` tool | `create_memory_entry(store, actor, path, content)` | `POST /api/agents/v1/memory-stores/{id}/entries` |
| `recall` tool | `search_memory_entries(store, actor, query)` | `POST …/entries:search` |

## End-to-end commands

```bash
# 1. Provision the two stores (retry memory create — see caveats)
mason -p ml_inference_staging memory   stores create --display-name my-agent-mem
mason -p ml_inference_staging sessions stores create --name my-agent-sessions

# 2. Run the agent (the code above); it creates sessions + reads/writes both stores.

# 3. Inspect what persisted
mason -p ml_inference_staging sessions items   list   --store my-agent-sessions --session-id <sid>
mason -p ml_inference_staging memory   entries list   --store <mem-id> --actor-id <actor>
mason -p ml_inference_staging memory   entries search --store <mem-id> --actor-id <actor> --query "terse"
```

## Caveats & findings (as of 2026-08-18, staging)

- **`strict` shim is Claude-only.** Databricks' Anthropic serving path rejects the `strict`
  field the SDK always emits (`400 tools.0.custom.strict: Extra inputs are not permitted`);
  the OpenAI-model path accepts it. Verified `openai-agents` runs against `databricks-gpt-5`
  with **no shim** at all.
- **Not every RPC is implemented.** Works: session store create/get, `create_session`,
  `items:append`, `items` list; memory store create/get/list/delete and entry
  create/get/list/search/delete. UNIMPLEMENTED (surface a clean `NOT_IMPLEMENTED`): session
  store list/update/delete; `list/get/update/delete/fork` session; `items:pop`/`items:clear`;
  memory store/entry `update`. So the session backend's `pop_item`/`clear_session` are
  best-effort no-ops, and a fresh session must keep its own `session_id` (you can't list/get it back).
- **Memory-store creation is flaky on staging** (Lakebase provisioning). `CreateManagedMemoryStore`
  can return `DEADLINE_EXCEEDED` or `INTERNAL_ERROR: Failed to initialize Lakebase schema: database
  "memory-…" does not exist` and leave a half-provisioned store (metadata row exists, entries
  table missing). Retry with a fresh name until create returns clean; entries work reliably on a
  cleanly-provisioned store. Session-store creation is unaffected.
- **Auth inside a deployed Databricks App.** For a *local* run, the profile bearer above is fine.
  In a *deployed* App the SDK authenticates the app's service principal via OAuth, so
  `config.token` is empty — take the bearer from `w.config.authenticate()["Authorization"]`
  (as above) rather than `config.token`, or the OpenAI client raises `Missing credentials`.
```
