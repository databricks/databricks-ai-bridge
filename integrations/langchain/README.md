# 🦜🔗 Using Databricks AI Bridge with Langchain

Integrate Databricks AI Bridge package with Langchain to allow seamless usage of Databricks AI features with Langchain/Langgraph applications.

## Installation

```sh
# install from the source
pip install git+ssh://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/langchain
```

> [!NOTE]
> Once this package is published to PyPI, users can install via `pip install databricks-langchain`

## Get started

### (Preview) Use a Genie space as an agent

> [!NOTE]
> Requires Genie API Private Preview. Reach out to your account team for enablement. 

```python
from databricks_langchain.genie import GenieAgent

genie_agent = GenieAgent("space-id", "Genie", description="This Genie space has access to sales data in Europe")
```
