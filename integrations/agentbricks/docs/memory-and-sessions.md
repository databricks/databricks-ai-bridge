# Memory and sessions

An agent needs two kinds of state: the state of the interaction it is handling right now, and the
durable knowledge it carries from one conversation to the next. Databricks provides a managed store
for each, both backed by Lakebase and usable from agents built on any framework:

- **Managed agent sessions** store an agent's session state for one interaction. Most commonly this
  is the conversation history — the ordered transcript of messages, tool calls, and results — but it
  can be any state a framework persists, such as a LangGraph graph. The agent reads it at the start
  of a turn and appends to it as the interaction runs.
- **Managed agent memory** stores durable facts, preferences, and decisions that an agent recalls in
  later, separate conversations through text search.

![Sessions and memory: the agent reads and appends one conversation's transcript in the session store, and recalls and saves durable facts in the memory store, which outlive any single conversation.](sessions_and_memory.png)

The examples below use the [`AgentKitClient` Python SDK](#agentkit-sdk). The same operations are
available as `agentbricks sessions` and `agentbricks memory` CLI commands; see the [CLI command
reference](../cli.md) for the complete command set.

## AgentKit SDK

Use `AgentKitClient` from `databricks_agentkit` with an authenticated Databricks
`WorkspaceClient`, or omit the client to use the default Databricks SDK authentication. Its
`memory_stores` and `session_stores` collections create, get, and list stores. A returned store
manages its entries or sessions; returned memories and sessions own their `update()` and
`delete()` operations.

All `list()` methods return iterators that consume server pages automatically. List `page_size` and
search `limit` values must be between 1 and 100. `session.list_items()` also auto-pages.

## Sessions

A **session store** holds **sessions**, and each session holds an ordered list of **session items**.
A session is one interaction — typically a conversation thread — grouped under an `actor_id` (who it
belongs to; set this from trusted application context, never a model- or user-supplied value) and
identified by a caller-chosen `session_id` (the service generates one if you omit it). Each item is
an opaque, JSON-compatible `data` value — a message, tool call, result, or reasoning block — that
Databricks stores and returns verbatim, in order, and never mutates once appended.

Create a store, start a session, append the conversation's turns, and read the history back on a later
request:

```python
from databricks.sdk import WorkspaceClient
from databricks_agentkit import AgentKitClient

agentkit = AgentKitClient(WorkspaceClient())

session_store = agentkit.session_stores.create("support-agent-sessions")
session = session_store.add(actor_id="customer-123", session_id="case-456")

session.append_items(
    [
        {"type": "message", "role": "user", "content": "I need help with my cluster."},
        {"type": "message", "role": "assistant", "content": "Let's take a look."},
    ]
)

# On a later turn, reload the session and read its full history in order.
session = session_store.get("case-456")
history = [item.data for item in session.list_items()]  # list_items auto-pages
```

A session can be **forked** into an independent branch: a new session seeded with the original's
history, linked back to its origin by `parent_session_id`. Fork the full history, or only up to a
specific item, to explore an alternate continuation without disturbing the original thread:

```python
branch = session.fork(actor_id="customer-123")  # add up_to_item_id=... to branch up to one item
```

Deleting a session that has such descendants requires `session.delete(force=True)` to cascade.

## Memory

A **memory store** holds **memory entries**. Each entry is a free-form `content` string plus a short
`description` used for retrieval, keyed by three fields: `actor_id` (whose memory it is — set from
trusted application context, never a model- or user-supplied value), `path` (a filesystem-like key
within an actor, such as `/preferences/response-style.md`), and an optional `session_id` (the session
an entry came from, for provenance). An entry is uniquely identified by its `actor_id`, `path`, and
optional `session_id`.

Write an entry when the agent learns something durable, then recall it in a later, separate
conversation with a natural-language search. Results are ranked by full-text (BM25) relevance, up to
100 entries, with no pagination or vector similarity:

```python
from databricks.sdk import WorkspaceClient
from databricks_agentkit import AgentKitClient

agentkit = AgentKitClient(WorkspaceClient())

memory_store = agentkit.memory_stores.create("support-agent-memory")
memory_store.add(
    actor_id="user-123",
    path="/preferences/communication.md",
    content="Prefers email over phone. Timezone: PST.",
    description="User 123 communication preferences",
)

# In a later, separate conversation, recall what the agent knows about this user.
results = memory_store.search(actor_id="user-123", query="communication preferences", limit=10)
```

To browse rather than search, `memory_store.list(actor_id=..., path_prefix=...)` returns entries
directly.

> **`actor_id` partitions data; it is not access control.** Both stores are workspace-scoped and
> authorized at the store level, so any principal that can reach a store can read and write every
> actor's entries. For strict isolation between tenants or users, use a separate store per boundary.
> Grant another principal — such as your app's service principal — access with
> `session_store.grant_permission(principal_id)` or `memory_store.grant_permission(principal_id)`;
> `agentbricks deploy` attempts this grant for the deployed app.

## Framework adapters

In an agent configured with `server = "agentbricks"`, the framework adapter reads and appends
session state for you. The adapter resolves a bound store when one is configured. During local
`agentbricks dev`, sessions use process-local state and long-term memory is off; a deployed agent
gets the managed stores through the environment configured by `agentbricks deploy`.

### LangGraph

Pass `checkpointer()` when you build the agent and scope each run with `thread_config(session_id)`.
Pass the trusted actor as the second argument when the graph's state must be partitioned by actor.

```python
from databricks_agentkit.langgraph import checkpointer, thread_config

agent = create_agent(model=..., tools=[...], checkpointer=checkpointer())
result = await agent.ainvoke(inputs, config=thread_config(session_id))
```

Add the memory tools to the model and execution tool list. `memory_tools(actor)` exposes `remember`
and `recall` bound to one actor's partition. It resolves the store from the `[memory_store]` binding,
carried to the runtime by the `AGENT_MEMORY_STORE` environment variable that `agentbricks deploy`
injects. It returns no tools when no store is set, so the agent runs unchanged:

```python
from databricks_agentkit.langgraph import memory_tools

agent = create_agent(model=..., tools=[*your_tools, *memory_tools(actor)])
```

### OpenAI Agents SDK

The OpenAI Agents adapter exposes the same session and memory capabilities. Pass
`session_store(session_id)` to `Runner.run` and add `memory_tools(actor)` to the agent's tools. The
optional `actor` argument partitions durable session and memory data; derive it from trusted
application context.

```python
from agents import Agent, Runner
from databricks_agentkit.openai import memory_tools, session_store

agent = Agent(model=..., tools=[*your_tools, *memory_tools(actor)])
result = await Runner.run(
    agent,
    messages,
    session=session_store(session_id, actor),
)
```

Both adapters resolve their store from the explicit argument first, then the corresponding
`AGENT_SESSION_STORE` or `AGENT_MEMORY_STORE` environment variable, and then the `agent.toml`
binding. With no managed store configured, the session helpers keep state in process and the memory
helper returns no tools. This is the local development behavior; deployment supplies the managed
store configuration.

## Declaring and provisioning stores

For a deployed agent, `agent.toml` declares which stores it uses and `agentbricks deploy` provisions
them — you do not create stores by hand for the deployed project. `agentbricks init` declares a
default memory and session store named from the project; override those names, point at stores you
already have, or let `deploy` create them:

To scaffold a project with default memory and session stores declared in `agent.toml`:

```sh
agentbricks init my-agent
```

To scaffold with specific store names instead:

```sh
agentbricks init my-agent --memory-store support-agent-memory --session-store support-agent-sessions
```

For an existing project, bind the stores in `agent.toml`, then deploy. Binding edits the manifest;
it does not create the remote stores:

```sh
# Run these commands from the project directory.
cd my-agent
agentbricks sessions bind support-agent-sessions
agentbricks memory bind support-agent-memory

# deploy creates any declared-but-missing store and attempts the App service-principal grant.
agentbricks deploy my-agent
```

Memory and session stores are independent resources: deleting one never affects the other. For runtime
hooks, endpoint behavior, and recovery, see the [runtime guide](../src/databricks_agentkit/runtime/README.md).
