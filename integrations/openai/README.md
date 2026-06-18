#  Databricks OpenAI Integration

The `databricks-openai` package provides seamless integration of Databricks AI features into OpenAI applications.

## Installation

### From PyPI
```sh
pip install databricks-openai
```

### From Source
```sh
pip install git+https://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/openai
```

## Key Features

- **Vector Search:** Store and query vector representations using `VectorSearchRetrieverTool`.
- **OpenAI-compatible clients:** Use Databricks authentication with OpenAI SDK resources,
  including optional separate routing for Conversations API calls.

## Getting Started

### Use Vector Search on Databricks
```python
# Step 1: call model with VectorSearchRetrieverTool defined
dbvs_tool = VectorSearchRetrieverTool(index_name="catalog.schema.my_index_name")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "Using the Databricks documentation, answer what is Spark?"
    }
]
first_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=[dbvs_tool.tool]
)

# Step 2: Execute function code – parse the model's response and handle function calls.
tool_call = first_response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = dbvs_tool.execute(query=args["query"])  # For self-managed embeddings, optionally pass in openai_client=client

# Step 3: Supply model with results – so it can incorporate them into its final response.
messages.append(first_response.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result)
})
second_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)
```

### Use Conversations API state

Conversations API calls default to `{workspace_url}/api/2.1/unity-catalog`, even when
Responses API calls use another Databricks OpenAI-compatible base URL such as AI Gateway.
Use `conversations_base_url` only when the Conversations API is served from a custom path.

```python
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

workspace_client = WorkspaceClient()

client = DatabricksOpenAI(
    workspace_client=workspace_client,
    use_ai_gateway=True,
)

conversation = client.conversations.create()
response = client.responses.create(
    model="databricks-claude-sonnet-4-5",
    conversation=conversation.id,
    input="Tell me about Databricks",
)
```

---

## Contribution Guide
We welcome contributions! Please see our [contribution guidelines](https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/langchain) for details.

## License
This project is licensed under the [MIT License](LICENSE).

Thank you for using Databricks OpenAI!
