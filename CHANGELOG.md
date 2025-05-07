# CHANGELOG

## databricks-ai-bridge 0.4.2, databricks-langchain 0.4.2, databricks-openai 0.3.2 (2025-05-07)

### Improvements
- Allow backticked vector search index names
- Include embedding model as a resource for VectorSearchRetrieverTool
- Support setting retriever schema (doc_uri, chunk_id) for VectorSearchRetrieverTool
- Support for passing filters at query time and forwarding arbitrary kwargs through VectorSearchRetrieverTool

## databricks-ai-bridge 0.4.1, databricks-langchain 0.4.1, databricks-openai 0.3.1 (2025-03-27)

### Improvements
- Fix Databricks Connect and Tabulate Dependencies
- Remove default temperature for ChatDatabricks

## databricks-ai-bridge 0.4.0, databricks-langchain 0.4.0, databricks-openai 0.3.0 (2025-03-10)

### Highlights
- Support On Behalf Of User rights with genie and VectorSearch Tool
- Update Genie Integration for PuPr APIs

### Improvements
- Improve `VectorSearchRetrieverTool` by disabling unwanted notices
- Updated documentation for installing integration packages
- Update inline unitycatalog imports
- Better validation of VectorSearchRetrieverTools

## 0.3.0 (2025-02-07)

### Highlights
- Adding integration with databricks-openai
- Introducing [api docs](https://api-docs.databricks.com/python/databricks-ai-bridge/index.html)

### Improvements
- Improving `VectorSearchRetrieverTool` API in both langchain and openai by introducing automatic description and resources needed to log the model
- Introducing model as an argument to be passed to ChatDatabricks in Langchain and deprecating notice endpoint to confer with the schema in Langchain.


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
