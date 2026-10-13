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

With `databricks-langchain>=0.22.0`, configure the model before passing it to a LangGraph agent:

```python
llm = ChatDatabricks(
    model="system.ai.claude-sonnet-4-5",
    use_ai_gateway=True,
    prompt_caching="auto",
)
```

The shared Databricks client adds explicit cache points to system instructions and recent
user/tool-result boundaries on every Chat Completions request. The system prefix includes tool
definitions. It leaves stored history untouched and uses up to four five-minute ephemeral cache
points. Caller-supplied `cache_control` fields take precedence. For a custom-named Claude endpoint,
set `prompt_cache_provider="anthropic"`. Other model providers are unchanged.

The library default is `"off"`; Agent Bricks templates enable `"auto"`. Cache eligibility depends on
the model's minimum prefix length, repeated content, and cache lifetime. Cache writes carry an
additional charge. Inspect `message.usage_metadata["input_token_details"]` for `cache_read` and
`cache_creation`; both are already included in `input_tokens`. Automatic placement requires Chat
Completions and cannot be combined with `use_responses_api=True`.

## Contribution Guide
We welcome contributions! Please see our [contribution guidelines](https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/langchain) for details.

## License
This project is licensed under the [MIT License](LICENSE).

Thank you for using Databricks LangChain!
