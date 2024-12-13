# 🦜🔗 Databricks LangChain Integration

The `databricks-langchain` package provides seamless integration of Databricks AI features into LangChain applications. This repository is now the central hub for all Databricks-related LangChain components, consolidating previous packages such as `langchain-databricks` and `langchain-community`.

## Installation

### From PyPI
```sh
pip install databricks-langchain
```

### From Source
```sh
pip install git+ssh://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/langchain
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

## Contribution Guide
We welcome contributions! Please see our [contribution guidelines](https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/langchain) for details.

## License
This project is licensed under the [MIT License](LICENSE).

Thank you for using Databricks LangChain!

