# 🦜🔗 Using Databricks AI Bridge with Langchain

Integrate Databricks AI Bridge package with Langchain to allow seamless usage of Databricks AI features with Langchain/Langgraph applications.

Note: This repository is the future home for all Databricks integrations currently found in langchain-databricks and langchain-community. Over time, we’ll be consolidating integrations here, including ChatDatabricks, DatabricksEmbeddings, DatabricksVectorSearch, and more.

Stay tuned for updates on when each integration becomes available in this package. For now, you can continue using langchain-databricks until the transition is complete.

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

### (Preview) Use a Genie space as an agent

> [!NOTE]
> Requires Genie API Private Preview. Reach out to your account team for enablement. 

```python
from databricks_langchain.genie import GenieAgent

genie_agent = GenieAgent("space-id", "Genie", description="This Genie space has access to sales data in Europe")
```
