# CHANGELOG

## databricks-langchain 0.12.0 (2025-12-16), databricks-openai 0.8.0 (2025-12-16), databricks-mcp 0.5.0 (2025-12-16), databricks-ai-bridge 0.11.0 (2025-12-16)

- LangChain MCP Server Adapters
- LangGraph DatabricksStore for long term memory on Databricks using Lakebase
- DCR Support for external UC Connections
- Single function and index MCP Servers for LangChain and OpenAI

## databricks-langchain 0.11.0 (2025-11-19)

- Addresses a dependency import bug introduced with the 0.10.0 release
- Adds support for passing conversation ID and `return_pandas` to return query results in `pandas` DataFrame format to `databricks_langchain.GenieAgent`

## databricks-ai-bridge 0.10.0 databricks-langchain 0.10.0 databricks-openai 0.7.0 (2025-11-18)

- Add support for LLM-generated filter parameters in VectorSearchRetrieverTool
- Support passing in Workspace Client to VSRetrieverTool
- Fix ChatModel Custom Outputs
- Vector Search add reranker parameter
- MCP Open AI Adapters
- Add Lakebase Pool Memory/Connection Pooling: LakebasePool and langchain CheckpointSaver

## databricks-ai-bridge 0.9.0 databricks-langchain 0.9.0 databricks-mcp 0.4.0 (2025-10-21)

- Langchain V1 compatability
- Improve OBO user authz error message with guidance on WorkspaceClient instantiation
- Higher visibility of Genie API's execution steps in MLflow traces
- Add OBO Debug Mode
- MCP: Fixed the mcp tools_calls which required MCP_Session_id which is obtained in the session via initialize call

## databricks-ai-bridge 0.8.1 databricks-langchain 0.8.2 (2025-10-13)

- Higher visibility of Genie API's execution steps in MLflow traces
- ChatDatabricks: turn on stream usage by default
- Use MMR, hybrid search and self-managed embeddings together in the langchain DatabricksVectorSearch object

## databricks-ai-bridge 0.6.0, databricks-langchain 0.6.0, databricks-openai 0.5.0, databricks-mcp 0.2.0 (2025-06-02)

### Improvements

- Add option to include similarity score in VectorSearchRetrieverTool result
- Introduce DatabricksMCPClient
- Include reasoning + SQL steps when Genie agent is queried
- Make ChatDatabricks compatible with ChatAgent and ResponsesAgent

## databricks-ai-bridge 0.5.1, databricks-langchain 0.5.1, databricks-openai 0.4.1 (2025-05-19)

### Bug Fixes

- Fix Databricks Vector Search Filter implementation
- Improve performance for Genie Truncation
- Fix Bug for Genie Description
- Deprecate python 3.9 as a supported environment

## databricks-ai-bridge 0.5.0, databricks-langchain 0.5.0, databricks-openai 0.4.0 (2025-05-12)

### Improvements

- Allow backticked vector search index names
- Include embedding model as a resource for VectorSearchRetrieverTool
- Support setting retriever schema (doc_uri, chunk_id) for VectorSearchRetrieverTool
- Support for passing filters at query time and forwarding arbitrary kwargs through VectorSearchRetrieverTool

## databricks-ai-bridge 0.4.2, databricks-langchain 0.4.3, databricks-openai 0.3.2 (2025-05-07) - YANKED

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
