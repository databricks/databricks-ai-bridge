# Databricks AI Bridge library

The Databricks AI Bridge library provides a shared layer of APIs to interact with Databricks AI features, such as [Databricks AI/BI Genie ](https://www.databricks.com/product/ai-bi/genie) and [Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html). Use these packages to help [author agents with Agent Framework](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent#requirements) on Databricks.

## Integration Packages

If you are using LangChain/LangGraph or the OpenAI SDK, we provide these integration packages for seamless integration of Databricks AI features.

- [`databricks-langchain`](./integrations/langchain/README.md) - For LangChain/LangGraph users
- [`databricks-openai`](./integrations/openai/README.md) - For OpenAI SDK users

## Installation

If you're using LangChain/LangGraph or OpenAI:

```sh
pip install databricks-langchain
pip install databricks-openai
```

For frameworks without dedicated integration packages:

```sh
pip install databricks-ai-bridge
```

## Long-running agents

`LongRunningAgentServer` adds Lakebase-backed background execution, heartbeat
recovery, result persistence, and optional SSE replay to MLflow ResponsesAgent
handlers. Recovery context can come from the server's event log or from an
agent-managed SDK session. Applications keep the normal `@invoke()` and
`@stream()` handlers and may register one optional `@on_resume()` request
transformer. The server calls it after claiming a stale attempt, then dispatches
the transformed request through the original invoke or stream mode.

See the [OpenAI Agents SDK examples](./examples/openai-sdk-agent/README.md) for
the same agent configured with framework-managed recovery, agent-session
recovery with SSE replay, and agent-session recovery without event-log
persistence.

```sh
pip install 'databricks-ai-bridge[agent-server]'
```

### Install from source

With https:

```sh
# For LangChain/LangGraph users (recommended):
pip install git+https://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/langchain
# For OpenAI users (recommended):
pip install git+https://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/openai
# Generic installation (only if needed):
pip install git+https://git@github.com/databricks/databricks-ai-bridge.git
```
