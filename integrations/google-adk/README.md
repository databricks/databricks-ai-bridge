# Databricks AI Bridge for Google ADK

This package provides Databricks AI integration for [Google Agent Development Kit (ADK)](https://github.com/google/adk-python), enabling you to use Databricks Vector Search and Genie in your ADK agents.

## Installation

```bash
pip install databricks-google-adk
```

## Features

- **VectorSearchRetrieverTool**: Search Databricks Vector Search indexes from ADK agents
- **GenieTool**: Query Databricks Genie AI/BI spaces using natural language
- **DatabricksToolset**: Bundle multiple Databricks tools together for easy agent configuration

## Quick Start

### Vector Search Tool

Use Databricks Vector Search in your ADK agent:

```python
from databricks_google_adk import VectorSearchRetrieverTool
from google.adk.agents import Agent
from google.adk.runners import Runner

# Create the Vector Search tool
vector_search = VectorSearchRetrieverTool(
    index_name="catalog.schema.my_index",
    num_results=5,
)

# Create an ADK agent with the tool
agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant that searches documents to answer questions.",
    tools=[vector_search.as_tool()],
)

# Run the agent
runner = Runner(agent=agent, app_name="search_app")
```

### Genie Tool

Query Databricks Genie for data insights:

```python
from databricks_google_adk import GenieTool
from google.adk.agents import Agent

# Create the Genie tool
genie = GenieTool(
    space_id="your-genie-space-id",
    tool_description="Ask questions about sales data",
)

# Create an ADK agent with the tool
agent = Agent(
    name="data_analyst",
    model="gemini-2.0-flash",
    instruction="You are a data analyst. Use the genie tool to answer questions about data.",
    tools=[genie.as_tool()],
)
```

### Using DatabricksToolset

Bundle multiple Databricks tools together:

```python
from databricks_google_adk import DatabricksToolset
from google.adk.agents import Agent

# Create a toolset with multiple tools
toolset = DatabricksToolset(
    vector_search_indexes=[
        "catalog.schema.products_index",
        "catalog.schema.docs_index",
    ],
    genie_space_ids=["genie-space-123"],
)

# Or build incrementally with method chaining
toolset = (
    DatabricksToolset()
    .add_vector_search_tool(
        index_name="catalog.schema.my_index",
        tool_name="search_products",
        tool_description="Search product catalog",
    )
    .add_genie_tool(
        space_id="genie-space-123",
        tool_name="ask_sales_data",
        tool_description="Query sales data",
    )
)

# Use with an ADK agent
agent = Agent(
    name="data_assistant",
    model="gemini-2.0-flash",
    instruction="You help users find products and analyze sales data.",
    tools=[toolset],
)
```

## Advanced Usage

### Self-Managed Embeddings

For Vector Search indexes with self-managed embeddings, provide an embedding function:

```python
from databricks_google_adk import VectorSearchRetrieverTool

def my_embedding_fn(text: str) -> list[float]:
    # Your embedding logic here
    return embeddings

vector_search = VectorSearchRetrieverTool(
    index_name="catalog.schema.self_managed_index",
    text_column="content",
    embedding_fn=my_embedding_fn,
)
```

### Dynamic Filters

Enable LLM-generated filters for more flexible querying:

```python
vector_search = VectorSearchRetrieverTool(
    index_name="catalog.schema.my_index",
    dynamic_filter=True,  # Allows LLM to generate filters
)
```

### Multi-turn Conversations with Genie

The GenieTool maintains conversation state for follow-up questions:

```python
from databricks_google_adk import GenieTool

genie = GenieTool(space_id="your-space-id")

# First question
result1 = genie.ask("What were total sales last month?")

# Follow-up question (uses same conversation)
result2 = genie.ask("Break that down by region")

# Start a new conversation
result3 = genie.ask("Show me top customers", new_conversation=True)

# Reset conversation state
genie.reset_conversation()
```

### Custom Authentication

Pass a custom WorkspaceClient for authentication:

```python
from databricks.sdk import WorkspaceClient
from databricks_google_adk import VectorSearchRetrieverTool, GenieTool

# Create a workspace client with custom configuration
client = WorkspaceClient(
    host="https://your-workspace.databricks.com",
    token="your-token",
)

# Use with Vector Search
vector_search = VectorSearchRetrieverTool(
    index_name="catalog.schema.my_index",
    workspace_client=client,
)

# Use with Genie
genie = GenieTool(
    space_id="your-space-id",
    client=client,
)
```

## API Reference

### VectorSearchRetrieverTool

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | `str` | Name of the Vector Search index (format: `catalog.schema.index`) |
| `num_results` | `int` | Number of results to return (default: 5) |
| `columns` | `list[str]` | Columns to return in results |
| `filters` | `dict` | Static filters to apply to all searches |
| `dynamic_filter` | `bool` | Enable LLM-generated filters (default: False) |
| `tool_name` | `str` | Custom name for the tool |
| `tool_description` | `str` | Custom description for the tool |
| `text_column` | `str` | Text column name (required for self-managed embeddings) |
| `embedding_fn` | `Callable` | Embedding function (required for self-managed embeddings) |
| `workspace_client` | `WorkspaceClient` | Custom Databricks client |

### GenieTool

| Parameter | Type | Description |
|-----------|------|-------------|
| `space_id` | `str` | Genie space ID |
| `tool_name` | `str` | Custom name for the tool (default: "ask_genie") |
| `tool_description` | `str` | Custom description for the tool |
| `client` | `WorkspaceClient` | Custom Databricks client |
| `return_pandas` | `bool` | Return results as pandas DataFrames (default: False) |

### DatabricksToolset

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector_search_indexes` | `list[str]` | List of Vector Search index names |
| `genie_space_ids` | `list[str]` | List of Genie space IDs |
| `workspace_client` | `WorkspaceClient` | Custom Databricks client |
| `embedding_fn` | `Callable` | Embedding function for self-managed indexes |
| `tool_filter` | `list[str]` | Filter tools by name |
| `tool_name_prefix` | `str` | Prefix to add to all tool names |

## Requirements

- Python >= 3.10
- google-adk >= 1.0.0
- databricks-ai-bridge >= 0.4.0
- databricks-vectorsearch >= 0.40

## License

Apache-2.0
