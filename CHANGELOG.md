# CHANGELOG

## 0.0.3 (2024-11-12)
This is a patch release that includes bugfixes.

Bug fixes:

- Update Genie API polling logic to account for COMPLETED query state (#16, @prithvikannan)


## 0.0.2 (2024-11-01)
Initial version of databricks-ai-bridge and databricks-langchain packages

Features:

- Support for Databricks AI/BI Genie via the `databricks_langchain.GenieAgent` API in `databricks-langchain`
- Support for most functionality in the existing `langchain-databricks` under `databricks-langchain`. Specifically, this 
  release introduces `databricks_langchain.ChatDatabricks`, `databricks_langchain.DatabricksEmbeddings`, and
  `databricks_langchain.DatabricksVectorSearch` APIs. 
