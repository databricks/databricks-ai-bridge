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

## Durable Runtime

[`DatabricksDurableRuntime`](./src/databricks_ai_bridge/durable_runtime/README.md)
adds Lakebase-backed request/response persistence, heartbeat detection, and
stale-attempt recovery around a caller-owned async handler. The handler remains
responsible for agent sessions and checkpoints.

[`DatabricksDurableAgentServer`](./src/databricks_ai_bridge/durable_agent_server/README.md)
extends MLflow AgentServer with blocking, background, retrieval, and stale
recovery while preserving its `@invoke()` handler contract.

See the [OpenAI Agents SDK App](./examples/openai-sdk-agent/README.md) for a
complete FastAPI, background polling, SDK session, and Databricks Apps example.

```sh
pip install 'databricks-ai-bridge[memory]'
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
