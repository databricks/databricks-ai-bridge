# Migrate from DatabricksStore to managed memory

This guide covers moving an agent from the `DatabricksStore` in `databricks-langchain`, which keeps memories in a Lakebase table you own, to the managed memory store behind `memory_store()`. There is no data export or import tool. Existing memories are not copied automatically; the [data section](#data-migration) outlines a copy you script yourself.

## Why migrate

- **Simpler deployment**: No Postgres connection and no Lakebase project to manage. The managed memory store is reached through the AgentKit client, the same way sessions are.
- **Provisioned by deploy**: A `[memory_store]` binding in agent.toml creates the store on `agentbricks deploy`, and deploy grants the app access.

## Before you start

- Review the limits in the [LangGraph memory store section](../README.md#langgraph-memory-store-as-a-basestore) of the README: the 100-entry search window, last-writer-wins writes, and TTL that counts from the last write and is never refreshed by reads.
- **Recall may change.** `DatabricksStore` ranks by your embedding endpoint only when you configured one (`embedding_endpoint` or `embeddings`). Without one, its search ignores the query and returns rows newest first, with no score. The managed store ranks with the service, which the managed memory docs describe as full-text (BM25) combined with semantic matching over content and description; the README has a short check for whether semantic matching is on in your workspace. Measure recall on your own data before cutover. The managed store has no client-side embeddings and no custom index fields; if you need to control the embedding model, stay on `DatabricksStore`.
- Existing memories are not migrated. Decide before cutover whether to script a copy or accept the loss.

## Migration steps

### 1. Remove the Postgres connection

Remove the code that builds the `DatabricksStore` and the Postgres or Lakebase connection it uses, such as the `psycopg` connection, SQLAlchemy engine, or `LakebasePool`. Then create the store:

```python
from databricks_agentkit.langgraph import memory_store

store = memory_store()  # store id: store= argument, then AGENT_MEMORY_STORE, on first use
```

### 2. Update imports

Replace the `databricks-langchain` store and checkpointer imports (`DatabricksStore`, `AsyncDatabricksStore`, `CheckpointSaver`, `AsyncCheckpointSaver`) with the AgentKit ones:

```python
from databricks_agentkit.langgraph import checkpointer, memory_store, thread_config
```

### 3. Wire the store into the graph

Build the store once, at startup, without `actor=`. Pass it to the agent with `store=`, the same way you pass the checkpointer. Putting the store in graph state does not work. Pass the user on every request, through `thread_config`:

```python
from databricks_agentkit.langgraph import checkpointer, memory_store, thread_config
from langchain.agents import create_agent

store = memory_store()  # once, at startup: no actor= here

agent = create_agent(model, tools=tools, checkpointer=checkpointer(), store=store)

config = thread_config(session_id, actor=user_id)  # per request
result = await agent.ainvoke({"messages": inputs}, config=config)
```

Inside a node, call `get_store()` from `langgraph.config`. Compile the store on the top-level graph; subgraphs inherit it from the run, but a store set only on a subgraph is not seen by its nodes.

Do not build a store with `actor=` and reuse it for every user. Such a store is pinned to that actor, and a run for a different user raises.

### 4. Update agent.toml

Declare the memory store binding so deploy creates it and grants the app access:

```toml
[memory_store]
name = "my-agent-memory"
```

If the store already exists, bind it instead:

```sh
agentbricks memory bind existing-store-name
```

### 5. Remove custom Postgres and Lakebase setup

Delete any deploy script or setup code that creates a Lakebase project, creates schemas, or manages Postgres connections for memory. The managed memory store handles provisioning. Once the binding from step 4 is in place, deploy is:

```sh
agentbricks deploy my-agent
```

### 6. Take the actor from the trusted identity

The actor is the signed-in user. Read it from the identity the Apps ingress forwards, never from a request body field. The bundled UI reads `x-forwarded-email`, then `x-forwarded-user`:

```python
from collections.abc import Mapping

FORWARDED_USER_HEADERS = ("x-forwarded-email", "x-forwarded-user")


def actor_from_headers(headers: Mapping[str, str]) -> str:
    """The signed-in user, from headers set by the Apps ingress. Never the request body."""
    for name in FORWARDED_USER_HEADERS:
        if value := headers.get(name):
            return value
    raise PermissionError("no forwarded user identity on the request")


user_id = actor_from_headers({"x-forwarded-email": "user@example.com"})
```

The bundled agent-langgraph template still reads `actor` from the request payload (`runtime/adapter.py`). Replace that with the forwarded identity before the app serves real users. A resumed run, such as one continuing after an approval, needs the same actor as the original run.

### 7. Test locally

`agentbricks dev` removes the memory binding from the manifest it runs, but it passes your shell environment through, so an `AGENT_MEMORY_STORE` set in the shell reaches the app:

```sh
AGENT_MEMORY_STORE=<store-id> agentbricks dev
```

Treat this as a local override: the dev help still says memory is off. The variable takes the store id, not the display name. A local run with it set reads and writes that live store with your own credentials, so use a dev store. Without a store id, the store is an in-process stand-in that keeps nothing across restarts.

### 8. Deploy and check the grant

```sh
agentbricks deploy my-agent
```

Deploy creates the declared store and grants the app's service principal access. The grant is best-effort: if it fails, deploy still succeeds and prints a next step. Check the deploy output before the first memory call. If the grant failed, run the grant shown in the [README section](../README.md#langgraph-memory-store-as-a-basestore) before use. A deployed App with no store bound raises on its first memory call rather than falling back to the in-process stand-in.

## Data migration

Memories in the `DatabricksStore` table and in the managed store are not compatible, and nothing copies them for you. There is no export or import command. `agentbricks memory entries create` writes one entry at a time, so it is not a bulk copy route. Its `--content` is raw text; for the store to read an entry, the content must be the store's envelope JSON, the path must be under `/langgraph/`, and the actor must be the storage actor (`<user-id>:langgraph`).

If old memories matter, script a one-time copy:

1. **Export read-only.** Connect to the legacy Lakebase with a read-only role. Read `prefix`, `key`, `value`, `created_at`, `updated_at`, `expires_at` and `ttl_minutes` from the table `store`, in the schema the legacy store was built with (`schema=`, default `public`). These names match langgraph-checkpoint-postgres 3.2.0; older 2.x releases may lack the TTL columns, so check them against your version first. Write the export to a private directory, outside the repo.
2. **Map rows to actors.** The namespace is the prefix split on dots, unchanged, so existing `aget` and `asearch` calls still hit the same tuples. The legacy table has no actor column, so use a mapping file that says which label is the user id. Rows with no matching rule go to a report, not the store.
3. **Decide per row.** Skip expired rows. For a row with a TTL, pass `ttl=` with its remaining minutes if they are above zero, or skip it. Skip values that aren't JSON objects, since the store holds JSON objects only.
4. **Write through the public API.** Write each row with a store pinned to its mapped actor, and pass the store id explicitly. A script may pin an actor outside a run. The store writes under the storage actor `<actor>:langgraph`, so a CLI check must use that id too, for example `agentbricks memory entries list --store <store-id> --actor-id <user-id>:langgraph --path-prefix /langgraph/`:

```python
from databricks_agentkit.langgraph import memory_store

STORE_ID = "<store-id>"  # required: without it memory_store() falls back to an in-process stand-in


async def import_row(actor: str, namespace: tuple[str, ...], key: str, value: dict) -> bool:
    if not isinstance(value, dict):
        return False  # the store holds JSON objects only
    store = memory_store(STORE_ID, actor=actor)  # scripts pin one actor per write
    if await store.aget(namespace, key) is not None:
        return False  # the app or an earlier run already wrote this key
    await store.aput(namespace, key, value)
    item = await store.aget(namespace, key)
    return item is not None and item.value == value


await import_row("user-123", ("preferences",), "communication", {"style": "terse"})
```

Pass the store id explicitly. Without one, `memory_store` returns an in-process stand-in that keeps nothing, and a read-back check against it still passes.

5. **Make re-runs safe.** Skip a row whose key already exists in the target, as `import_row` does; that covers app writes made after cutover and rows from an earlier run. Freeze legacy writes during the window. Timestamps reset to import time, since the managed store keeps its own.
6. **Verify by reading back.** Count the source rows against written, skipped (by reason), and unmapped. Read each written row with `aget` and compare it with `item.value == value`. Do not verify with search, because search results depend on ranking.
7. **Keep a manifest.** Record a hash, path, actor, and status per row, with no values. Rollback can then delete only what the script created, and only where the content still matches.

## Rollback

To roll back to `DatabricksStore`, redeploy the old code branch and re-establish its Postgres or Lakebase connection. Entries written to the managed store after cutover stay there, since nothing copies them back to the legacy table. The copy in the data section is one-way.
