"""
Demo showing the filter parameter schema and examples for VectorSearchRetrieverTool.

This script demonstrates the filter parameter structure without needing to connect to a real index.
"""

import json
from databricks_ai_bridge.vector_search_retriever_tool import FilterItem, VectorSearchRetrieverToolInput

print("="*80)
print("VectorSearchRetrieverTool Filter Parameter Documentation")
print("="*80)

# Show the FilterItem schema
print("\n1. FilterItem Schema:")
print("-" * 80)
print(json.dumps(FilterItem.model_json_schema(), indent=2))

# Show the input schema
print("\n2. VectorSearchRetrieverToolInput Schema:")
print("-" * 80)
input_schema = VectorSearchRetrieverToolInput.model_json_schema()
print(json.dumps(input_schema['properties']['filters'], indent=2))

# Example filter structures
print("\n3. Example Filter Structures:")
print("-" * 80)

examples = [
    {
        "description": "Simple equality filter",
        "filters": [{"key": "category", "value": "electronics"}]
    },
    {
        "description": "Multiple values (OR within same column)",
        "filters": [{"key": "category", "value": ["electronics", "computers"]}]
    },
    {
        "description": "Exclusion filter",
        "filters": [{"key": "status NOT", "value": "archived"}]
    },
    {
        "description": "Comparison filters (range)",
        "filters": [
            {"key": "price >=", "value": 100},
            {"key": "price <", "value": 500}
        ]
    },
    {
        "description": "Pattern matching",
        "filters": [{"key": "description LIKE", "value": "wireless"}]
    },
    {
        "description": "OR logic across columns",
        "filters": [{"key": "category OR subcategory", "value": ["tech", "gadgets"]}]
    },
    {
        "description": "Complex combination",
        "filters": [
            {"key": "category", "value": "electronics"},
            {"key": "price >=", "value": 50},
            {"key": "price <", "value": 200},
            {"key": "status NOT", "value": "discontinued"},
            {"key": "brand", "value": ["Apple", "Samsung", "Google"]}
        ]
    }
]

for i, example in enumerate(examples, 1):
    print(f"\n{i}. {example['description']}:")
    print(json.dumps(example['filters'], indent=2))

# Show how LLM would receive this in tool description
print("\n4. How this appears in OpenAI tool schema:")
print("-" * 80)

# Simulate what would be in the tool definition
tool_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The string used to query the index"
        },
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The filter key, which includes the column name and can include operators like 'NOT', '<', '>=', 'LIKE', 'OR'"
                    },
                    "value": {
                        "description": "The filter value, which can be a single value or an array of values"
                    }
                },
                "required": ["key", "value"]
            },
            "description": "Optional filters to refine vector search results... (with examples)"
        }
    }
}

print(json.dumps(tool_schema, indent=2))

# Example of LLM-generated filters
print("\n5. Example LLM-generated filters for different queries:")
print("-" * 80)

llm_examples = [
    {
        "user_query": "Find documentation about Unity Catalog from 2024",
        "llm_generated_filters": [
            {"key": "product", "value": "Unity Catalog"},
            {"key": "year >=", "value": 2024}
        ]
    },
    {
        "user_query": "Show me machine learning tutorials that are not archived",
        "llm_generated_filters": [
            {"key": "topic", "value": "machine learning"},
            {"key": "type", "value": "tutorial"},
            {"key": "status NOT", "value": "archived"}
        ]
    },
    {
        "user_query": "Find recent SQL or Python documentation",
        "llm_generated_filters": [
            {"key": "language OR topic", "value": ["SQL", "Python"]},
            {"key": "updated_date >=", "value": "2024-01-01"}
        ]
    }
]

for i, example in enumerate(llm_examples, 1):
    print(f"\n{i}. User Query: \"{example['user_query']}\"")
    print("   LLM generates:")
    print(f"   {json.dumps(example['llm_generated_filters'], indent=2)}")

print("\n" + "="*80)
print("Key Points:")
print("="*80)
print("1. Filters are an array of key-value pairs")
print("2. Keys can include operators: NOT, <, <=, >, >=, LIKE, OR")
print("3. Values can be single values or arrays (for multiple values)")
print("4. LLMs can generate these filters based on natural language queries")
print("5. The filter description includes available columns when possible")
print("="*80)
