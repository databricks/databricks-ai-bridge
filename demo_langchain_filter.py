"""
Demo script showing LLM-generated filter parameters with LangChain's VectorSearchRetrieverTool.

This demonstrates:
1. Creating a VectorSearchRetrieverTool with the dogfood profile
2. Using it with a LangChain agent to answer questions with filters
3. Showing how the LLM generates appropriate filter parameters
"""

import json
from databricks.sdk import WorkspaceClient
from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Setup
index_name = "ep.agent_demo.product_docs_index"
model_name = "databricks-meta-llama-3-3-70b-instruct"

# Create WorkspaceClient with the dogfood profile
print("Creating WorkspaceClient with 'dogfood' profile...")
workspace_client = WorkspaceClient(profile='dogfood')
print(f"Connected to: {workspace_client.config.host}")
print(f"User: {workspace_client.current_user.me().user_name}")

# Create the vector search retriever tool with the workspace_client
print(f"\nCreating VectorSearchRetrieverTool for index: {index_name}")
retriever_tool = VectorSearchRetrieverTool(
    index_name=index_name,
    num_results=3,
    workspace_client=workspace_client,
    dynamic_filter=True
)

print(f"\nTool created: {retriever_tool.name}")
print(f"Tool description: {retriever_tool.description[:200]}...")

# Show the filter parameter schema
print("\n" + "="*80)
print("Filter Parameter Schema:")
print("="*80)
filter_schema = retriever_tool.args_schema.model_json_schema()
if 'properties' in filter_schema and 'filters' in filter_schema['properties']:
    print(json.dumps(filter_schema['properties']['filters'], indent=2)[:500] + "...")

# Create a ChatDatabricks model
print("\n" + "="*80)
print("Setting up LangChain Agent with ChatDatabricks")
print("="*80)

llm = ChatDatabricks(
    endpoint=model_name,
    target_uri=workspace_client.config.host + "/serving-endpoints"
)

# Create a simple prompt for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses vector search to find relevant product documentation. "
               "When searching, use filters to narrow down results based on the user's requirements."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent
agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

# Example 1: Query that should trigger filters
print("\n" + "="*80)
print("Example 1: Query with implicit filter requirement")
print("="*80)

query1 = "Find documentation about Data Engineering products"
print(f"\nUser query: {query1}")
print("\nInvoking agent...")

try:
    result1 = agent_executor.invoke({"input": query1})
    print(f"\nAgent response: {result1['output']}")
except Exception as e:
    print(f"Error: {e}")

# Example 2: Direct tool invocation with manual filters
print("\n" + "="*80)
print("Example 2: Direct tool invocation with filters")
print("="*80)

manual_query = "workspace"
manual_filters = [
    {"key": "product_category", "value": "Data Engineering"}
]

print(f"\nQuery: {manual_query}")
print(f"Filters: {json.dumps(manual_filters, indent=2)}")

try:
    results = retriever_tool._run(query=manual_query, filters=manual_filters)
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results[:2], 1):
        print(f"\n{i}. Content: {doc.page_content[:200]}...")
        print(f"   Metadata: {json.dumps(doc.metadata, indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Example 3: Query with specific product category
print("\n" + "="*80)
print("Example 3: Agent query with specific category requirement")
print("="*80)

query3 = "Show me Databricks SQL documentation, filtering for Data Warehousing products"
print(f"\nUser query: {query3}")
print("\nInvoking agent...")

try:
    result3 = agent_executor.invoke({"input": query3})
    print(f"\nAgent response: {result3['output']}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("Demo complete!")
print("="*80)
