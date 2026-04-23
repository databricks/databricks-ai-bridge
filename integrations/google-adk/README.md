# Databricks AI Bridge for Google ADK

This package provides Databricks AI integration for [Google Agent Development Kit (ADK)](https://github.com/google/adk-python), enabling you to use Databricks Vector Search and Genie in your ADK agents.

## Installation

```bash
pip install databricks-google-adk
```

## Features

**Tools:**
- **VectorSearchRetrieverTool**: Search Databricks Vector Search indexes from ADK agents
- **GenieTool**: Query Databricks Genie AI/BI spaces using natural language

**Toolsets:**
- **DatabricksToolset**: Bundle multiple Databricks tools together for easy agent configuration
- **DatabricksMcpToolset**: Connect to Databricks MCP servers (UC Functions, Vector Search, Genie)

**Deployment:**
- **DatabricksAgentEngineApp**: Deploy Databricks-powered agents to Vertex AI Agent Engine
- **deploy_to_agent_engine**: One-step deployment helper function

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

### Databricks MCP Toolset

Connect to Databricks MCP servers for UC Functions, Vector Search, or Genie:

```python
from databricks_google_adk import DatabricksMcpToolset
from google.adk.agents import Agent

# Connect to UC Functions MCP server
toolset = DatabricksMcpToolset.for_uc_functions(
    catalog="my_catalog",
    schema="my_schema",
)

# Or connect to Vector Search MCP
toolset = DatabricksMcpToolset.for_vector_search(
    catalog="my_catalog",
    schema="my_schema",
)

# Or connect to Genie MCP
toolset = DatabricksMcpToolset.for_genie(
    space_id="my-genie-space-id",
)

# Use with an ADK agent
agent = Agent(
    name="function_caller",
    model="gemini-2.0-flash",
    instruction="You help users by calling Databricks functions.",
    tools=[toolset],
)
```

## Deployment to Vertex AI Agent Engine

Deploy your Databricks-powered ADK agents to Google Cloud's Vertex AI Agent Engine.

### Installation

```bash
pip install databricks-google-adk[deployment]
```

### Quick Deployment

```python
from databricks_google_adk import VectorSearchRetrieverTool
from databricks_google_adk.deployment import deploy_to_agent_engine
from google.adk.agents import Agent

# Create your agent
vector_search = VectorSearchRetrieverTool(index_name="catalog.schema.index")
agent = Agent(
    name="search_agent",
    model="gemini-2.0-flash",
    instruction="You help users search documents.",
    tools=[vector_search.as_tool()],
)

# Deploy to Agent Engine
remote_agent = deploy_to_agent_engine(
    agent=agent,
    project="my-gcp-project",
    location="us-central1",
    staging_bucket="gs://my-staging-bucket",
    databricks_host="https://my-workspace.databricks.com",
    databricks_token_secret="projects/my-project/secrets/databricks-token/versions/latest",
)

print(f"Deployed: {remote_agent.resource_name}")
```

### Step-by-Step Deployment

For more control, use the `DatabricksAgentEngineApp` class:

```python
from databricks_google_adk import DatabricksAgentEngineApp
import vertexai

# Create the deployable app
app = DatabricksAgentEngineApp(agent=agent)

# Test locally first
async for event in app.test_locally("Search for AI documents"):
    print(event)

# Initialize Vertex AI
client = vertexai.Client(project="my-project", location="us-central1")

# Get deployment config
config = app.get_deployment_config(
    staging_bucket="gs://my-bucket",
    databricks_host="https://workspace.databricks.com",
    databricks_token_secret="projects/my-project/secrets/db-token/versions/latest",
)

# Deploy
remote_agent = client.agent_engines.create(
    agent=app.adk_app,
    config=config,
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

### DatabricksMcpToolset

| Parameter | Type | Description |
|-----------|------|-------------|
| `server_url` | `str` | URL of the Databricks MCP server |
| `workspace_client` | `WorkspaceClient` | Custom Databricks client |
| `tool_filter` | `list[str]` | Filter tools by name |
| `tool_name_prefix` | `str` | Prefix to add to all tool names |

Factory methods:
- `DatabricksMcpToolset.for_uc_functions(catalog, schema)` - UC Functions MCP
- `DatabricksMcpToolset.for_vector_search(catalog, schema)` - Vector Search MCP
- `DatabricksMcpToolset.for_genie(space_id)` - Genie MCP

### deploy_to_agent_engine

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | The ADK Agent to deploy |
| `project` | `str` | Google Cloud project ID |
| `location` | `str` | Google Cloud region |
| `staging_bucket` | `str` | GCS bucket for staging (format: `gs://bucket`) |
| `databricks_host` | `str` | Databricks workspace URL |
| `databricks_token_secret` | `str` | Secret Manager secret for Databricks token |
| `display_name` | `str` | Display name for the deployed agent |
| `description` | `str` | Description for the deployed agent |

## Requirements

Core:
- Python >= 3.10
- google-adk >= 1.0.0
- databricks-ai-bridge >= 0.4.0
- databricks-vectorsearch >= 0.40
- databricks-mcp >= 0.5.0

For deployment to Agent Engine:
- google-cloud-aiplatform[agent_engines,adk] >= 1.112

## License

Apache-2.0
