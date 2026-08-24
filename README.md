# Databricks AI Bridge library

The Databricks AI Bridge library provides a shared layer of APIs to interact with Databricks AI features, such as [Databricks AI/BI Genie ](https://www.databricks.com/product/ai-bi/genie) and [Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html). Use these packages to help [author agents with Agent Framework](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent#requirements) on Databricks.

## Integration Packages

We provide integration packages for seamless use of Databricks AI features with supported frameworks and SDKs.

- [`databricks-langchain`](./integrations/langchain/README.md) - For LangChain/LangGraph users
- [`databricks-mason`](./integrations/mason/README.md) - For Mason users
- [`databricks-openai`](./integrations/openai/README.md) - For OpenAI SDK users

## Installation

Install the package for your framework or SDK:

```sh
pip install databricks-langchain
pip install databricks-mason
pip install databricks-openai
```

For frameworks without dedicated integration packages:

```sh
pip install databricks-ai-bridge
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
