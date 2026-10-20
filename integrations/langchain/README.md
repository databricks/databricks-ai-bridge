# 🦜🔗 Databricks LangChain Integration

The `databricks-langchain` package provides seamless integration of Databricks AI features into LangChain applications. This repository is now the central hub for all Databricks-related LangChain components, consolidating previous packages such as `langchain-databricks` and `langchain-community`.

## Installation

### From PyPI
```sh
pip install databricks-langchain
```

### From Source
```sh
pip install git+https://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/langchain
```

## Key Features

- **LLMs Integration:** Use Databricks-hosted large language models (LLMs) like Llama and Mixtral through `ChatDatabricks`.
- **Vector Search:** Store and query vector representations using `DatabricksVectorSearch`.
- **Embeddings:** Generate embeddings with `DatabricksEmbeddings`.
- **Genie:** Use [Genie](https://www.databricks.com/product/ai-bi/genie) in Langchain.

## Getting Started

### Use LLMs on Databricks
```python
from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")
```

### Use a Genie Space as an Agent (Preview)
> **Note:** Requires Genie API Private Preview. Contact your Databricks account team for enablement.

```python
from databricks_langchain.genie import GenieAgent

genie_agent = GenieAgent(
    "space-id", "Genie",
    description="This Genie space has access to sales data in Europe"
)
```

---

## Claude prompt caching

`ChatDatabricks` forwards caller-supplied controls through `extra_body` and content blocks.
For a Claude endpoint supporting native automatic placement (for example Sonnet 4.6):

```python
llm = ChatDatabricks(
    model="system.ai.claude-sonnet-4-6",
    use_ai_gateway=True,
    extra_params={"extra_body": {"cache_control": {"type": "ephemeral"}}},
)
```

The generic connector does not infer a caching policy. Agent Bricks templates use the separate
Claude adapter in `databricks_agentkit.claude`; see the [template caching guide](../agentbricks/docs/claude-prompt-caching.md)
for configuration, older endpoints, and manual controls.

Inspect `message.usage_metadata["input_token_details"]` for `cache_read` and `cache_creation`,
including streaming results. Both are already included in `input_tokens`. Cache eligibility and
pricing depend on the endpoint and repeated prefix.

## Contribution Guide
We welcome contributions! Please see our [contribution guidelines](https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/langchain) for details.

## License
This project is licensed under the [MIT License](LICENSE).

Thank you for using Databricks LangChain!
