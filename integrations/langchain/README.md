# 🦜🔗 Using Databricks AI Bridge with Langchain

Integrate Databricks AI Bridge package with Langchain to allow seamless usage of Databricks AI features with Langchain/Langgraph applications.

Note: This repository is the future home for all Databricks integrations currently found in `langchain-databricks` and `langchain-community`. We have now aliased `langchain-databricks` to `databricks-langchain`, consolidating integrations such as ChatDatabricks, DatabricksEmbeddings, DatabricksVectorSearch, and more under this package.

## Installation

### Install from PyPI
```sh
pip install databricks-langchain
```

### Install from source

```sh
pip install git+ssh://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/langchain
```

## Get started

### Use LLMs on Databricks

```python
from databricks_langchain import ChatDatabricks
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")
```

### (Preview) Use a Genie space as an agent

> [!NOTE]
> Requires Genie API Private Preview. Reach out to your account team for enablement. 

```python
from databricks_langchain.genie import GenieAgent

genie_agent = GenieAgent("space-id", "Genie", description="This Genie space has access to sales data in Europe")
```
