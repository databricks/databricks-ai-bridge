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
- **Sandbox:** Run an [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) agent's shell and filesystem tools inside a [Databricks sandbox](https://docs.databricks.com/aws/en/compute/serverless/sandbox) with `DatabricksSandboxClient` (install `databricks-openai[sandbox]`).

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

### Run an agent in a Databricks sandbox

Install the extra:

```sh
pip install databricks-openai[sandbox]
```

Register `DatabricksSandboxClient` as the sandbox backend on a `RunConfig`. The client provisions a Databricks sandbox when the run starts and tears it down when it finishes, so the agent's shell and filesystem tools execute inside the sandbox:

```python
from agents.run_config import RunConfig, SandboxRunConfig
from databricks_openai import DatabricksSandboxClient, DatabricksSandboxClientOptions

run_config = RunConfig(
    sandbox=SandboxRunConfig(
        client=DatabricksSandboxClient(profile="DEFAULT"),
        options=DatabricksSandboxClientOptions(),
    )
)

# Pass run_config to your Runner, e.g. Runner.run(agent, input, run_config=run_config)
```

`DatabricksSandboxClient(profile=...)` builds a `WorkspaceClient` from the named Databricks CLI profile; to reuse an existing client, pass `DatabricksSandboxClient(client=...)` instead.

---

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
