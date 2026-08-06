# Session Store SDK prototype

The experimental Session Store integration separates framework-specific adapters from one thin,
authenticated REST client:

- `DatabricksSessionStoreClient` owns Databricks authentication, REST calls, pagination, and
  Session/SessionEvent resource handling.
- `databricks_openai.agents.DatabricksSession` implements OpenAI Agents SDK's four-method
  `Session` protocol.
- `DatabricksClaudeSessionStore` implements Claude Agent SDK's transcript-mirroring
  `SessionStore` protocol, including subagent transcripts.

The default API prefix is `/api/2.0/agent-conversation`. It is a temporary prototype path and may
change with the REST contract. All three classes are marked experimental so their names and
constructor arguments may change before release.

## Install this branch

Install the core client and Claude adapter from the repository root:

```shell
pip install "databricks-ai-bridge @ git+https://github.com/annzhang-db/databricks-ai-bridge.git@codex/conversations-base-url"
```

Install the OpenAI integration package from its subdirectory:

```shell
pip install "databricks-openai @ git+https://github.com/annzhang-db/databricks-ai-bridge.git@codex/conversations-base-url#subdirectory=integrations/openai"
```

Claude applications also need the Claude Agent SDK version that provides `SessionStore`:

```shell
pip install "claude-agent-sdk>=0.2.128"
```

## OpenAI Agents SDK

```python
from agents import Agent, Runner
from databricks_openai.agents import DatabricksSession

session = DatabricksSession("conversation-123")
agent = Agent(name="Support agent", instructions="Help the customer.", model="...")

result = await Runner.run(
    agent,
    "Where is my order?",
    session=session,
)
```

`DatabricksSession` stores each native Agents SDK item verbatim in `SessionEvent.data`. It maps
`get_items`, `add_items`, `pop_item`, and `clear_session` directly to list, append, pop, and clear
REST operations. If `session_id` is omitted, the adapter creates one locally.

## Claude Agent SDK

```python
from claude_agent_sdk import ClaudeAgentOptions, query
from databricks_ai_bridge.session_store import DatabricksClaudeSessionStore

session_store = DatabricksClaudeSessionStore()

options = ClaudeAgentOptions(
    session_store=session_store,
    session_store_flush="batched",
    session_id="00000000-0000-0000-0000-000000000123",
)

async for message in query(prompt="Where is my order?", options=options):
    print(message)
```

Claude identifies storage by `project_key`, `session_id`, and an optional `subpath`. The adapter
maps the main transcript to one Databricks Session and each subagent `subpath` to a child Session.
This makes `list_subkeys` a direct-child listing and lets deletion of a main transcript use the
Session API's subtree cascade.

Claude retries transcript entries carrying a stable `uuid`, but the current SessionEvent contract
only exposes service-generated event IDs. The adapter performs a read-before-append duplicate
check for prototype testing. That check is not atomic under concurrent writers; a production
contract needs a caller idempotency key per event or equivalent conditional append semantics.

## LiteSwap

Supply the swap unit as the traffic ID without changing the API URL:

```python
client = DatabricksSessionStoreClient(
    traffic_id="testenv://liteswap/my-session-store",
)
```

You can also override `base_url` or set `DATABRICKS_SESSION_STORE_BASE_URL`. The traffic ID can be
set with `DATABRICKS_SESSION_STORE_TRAFFIC_ID`.
