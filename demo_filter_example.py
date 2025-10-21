"""
Demo script showing LLM-generated filter parameters with VectorSearchRetrieverTool.

This script demonstrates:
1. Creating a VectorSearchRetrieverTool with the product_docs_index
2. Using the tool with OpenAI to generate filters based on natural language queries
3. Showing how the LLM can automatically generate appropriate filter parameters
"""

import json
import os
from openai import OpenAI
from databricks_openai import VectorSearchRetrieverTool
from databricks.sdk import WorkspaceClient

# Setup
index_name = "ep.agent_demo.product_docs_index"
model = "databricks-meta-llama-3-3-70b-instruct"

# Create WorkspaceClient with the dogfood profile
print("Creating WorkspaceClient with 'dogfood' profile...")
workspace_client = WorkspaceClient(profile='dogfood')
print(f"Connected to: {workspace_client.config.host}")
print(f"User: {workspace_client.current_user.me().user_name}")

# Create the vector search retriever tool with the workspace_client
print(f"\nCreating VectorSearchRetrieverTool for index: {index_name}")
dbvs_tool = VectorSearchRetrieverTool(
    index_name=index_name,
    num_results=3,
    workspace_client=workspace_client,
    dynamic_filter=True
)

print(f"\nTool created: {dbvs_tool.tool['function']['name']}")
print(f"Tool description: {dbvs_tool.tool['function']['description'][:200]}...")

# Show the filter parameter schema
print("\n" + "="*80)
print("Filter Parameter Schema:")
print("="*80)
filter_param = dbvs_tool.tool['function']['parameters']['properties']['filters']
print(json.dumps(filter_param, indent=2))

# Show the full tool schema to inspect descriptions
print("\n" + "="*80)
print("Full Tool Schema (for inspection):")
print("="*80)
print(json.dumps(dbvs_tool.tool, indent=2))

# Create OpenAI client pointing to Databricks using the workspace_client's config
client = OpenAI(
    api_key=workspace_client.config.token,
    base_url=workspace_client.config.host + "/serving-endpoints"
)

# Let's also query the index to see what actual values exist for product_category
print("\n" + "="*80)
print("Sample data from the index (to see actual category values):")
print("="*80)
try:
    # Query without filters to see what's actually in the index
    sample_results = dbvs_tool.execute(
        query="product",
        openai_client=client
    )
    print(f"\nFound {len(sample_results)} sample results:")

    # Extract unique product categories
    categories = set()
    for doc in sample_results:
        # The doc content should have the category
        content = doc.get('page_content', '') or doc.get('content', '')
        # Try to extract category from the content
        if '<product_category>' in content:
            start = content.find('<product_category>') + len('<product_category>')
            end = content.find('</product_category>')
            if end > start:
                category = content[start:end]
                categories.add(category)

    if categories:
        print(f"\nActual product_category values found: {sorted(categories)}")
    else:
        print("\nCouldn't extract categories from sample results")
        print("\nFirst result structure:")
        if sample_results:
            print(json.dumps(sample_results[0], indent=2)[:500])
except Exception as e:
    print(f"Error fetching sample data: {e}")

# Example 1: Query that should trigger a filter
print("\n" + "="*80)
print("Example 1: Query with implicit filter requirement")
print("="*80)

messages = [
    {"role": "system", "content": "You are a helpful assistant that uses vector search to find relevant documentation."},
    {
        "role": "user",
        "content": "Find product documentation for Data Engineering products. Use filters to narrow down the results.",
    },
]

print(f"\nUser query: {messages[1]['content']}")
print("\nCalling LLM with tool...")

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=[dbvs_tool.tool],
    tool_choice="required"  # Force the model to use the tool
)

print("\nLLM Response:")
tool_call = response.choices[0].message.tool_calls[0] if response.choices[0].message.tool_calls else None

if tool_call:
    print(f"Tool called: {tool_call.function.name}")
    args = json.loads(tool_call.function.arguments)
    print(f"\nQuery: {args.get('query', 'N/A')}")
    print(f"Filters: {json.dumps(args.get('filters', []), indent=2)}")

    # Execute the tool
    print("\nExecuting vector search with filters...")
    try:
        results = dbvs_tool.execute(
            query=args["query"],
            filters=args.get("filters", None),
            openai_client=client
        )
        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {json.dumps(doc, indent=2)}")
    except Exception as e:
        print(f"Error executing tool: {e}")
else:
    print("No tool call made")

# Example 2: Manual filter specification
print("\n" + "="*80)
print("Example 2: Manual filter specification")
print("="*80)

manual_filters = [
    {"key": "product_category", "value": "Data Engineering"}
]

print(f"\nManual filters: {json.dumps(manual_filters, indent=2)}")
print("Executing search...")

try:
    results = dbvs_tool.execute(
        query="machine learning features",
        filters=manual_filters,
        openai_client=client
    )
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {json.dumps(doc, indent=2)[:200]}...")
except Exception as e:
    print(f"Error executing tool: {e}")

print("\n" + "="*80)
print("Demo complete!")
print("="*80)
