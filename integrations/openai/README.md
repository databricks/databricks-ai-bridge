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

---

## Claude prompt caching

With `databricks-openai>=0.18.0`, enable explicit cache points for Claude Chat Completions:

```python
from databricks_openai import AsyncDatabricksOpenAI

client = AsyncDatabricksOpenAI(use_ai_gateway=True, prompt_caching="auto")
# Pass this client to OpenAIChatCompletionsModel, or set_default_openai_client(client).
```

The same setting works on `DatabricksOpenAI` and Foundation Model API endpoints. The policy caches
system instructions (including the preceding tool definitions) and recent user/tool-result
boundaries. It runs on every model request, including tool loops, without modifying stored history.
Non-Claude models are unchanged. For a Claude endpoint with a custom name, also set
`prompt_cache_provider="anthropic"`.

The library default is `prompt_caching="off"`; Agent Bricks templates enable `"auto"`. Caller-supplied
`cache_control` fields take precedence: when present, automatic placement is skipped. The policy
uses up to four five-minute ephemeral cache points. Short prefixes below the model's minimum
length are not cached, and cache writes carry an additional charge. This policy applies to Chat
Completions; it does not add cache points to Responses requests.

Cache reads are available in `response.usage.prompt_tokens_details.cached_tokens`, and writes in
`response.usage.prompt_tokens_details.cache_write_tokens`, including streaming chunks. Native Claude
counters are retained. Databricks `prompt_tokens` already includes cached tokens; do not add them
again. See [prompt caching on Databricks](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#prompt-cache).

## Contribution Guide
We welcome contributions! Please see our [contribution guidelines](https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/langchain) for details.

### Cross-version compatibility tests

The `openai_cross_version_test` CI job installs current core with an unchanged historical
OpenAI integration. It runs the maintained tests in `tests/compatibility` rather than the
historical release's tests, so the test harness uses the tracing API supported by current core.
The 24 cases cover retriever construction, resources, retrieval results, and persisted traces.
Resource expectations retain the differences between historical releases.
External services are mocked; MLflow writes traces to a temporary local store.

To reproduce from the repository root, with a historical checkout at `older-version`:

```bash
pip install . older-version/integrations/openai --group dev
pip check
pytest -c pyproject.toml integrations/openai/tests/compatibility
```

## License
This project is licensed under the [MIT License](LICENSE).

Thank you for using Databricks OpenAI!
