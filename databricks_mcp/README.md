# Databricks MCP Library

The `databricks-mcp` package provides useful helpers to integrate MCP Servers in Databricks

## Installation

### From PyPI
```sh
pip install databricks-mcp
```

### From Source
```sh
pip install git+https://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=databricks_mcp
```

## Key Features

- **OAuth Provider**: Enables authentication across Databricks Notebooks, Model Serving, and local environments using the Databricks CLI.

## URL elicitation

An MCP server may ask the user to open a URL mid tool call (for example, to complete an auth flow). `DatabricksMCPClient` supports this, but **elicitation is off by default** — safe for agents embedded in headless or async runtimes (Model Serving, notebooks), where a stdin prompt or browser launch is meaningless and would block the event loop.

To opt in, pass an `elicitation_callback`. Agents should supply their own callback that surfaces the URL through their own channel (UI, response stream, approval card):

```python
from databricks_mcp import DatabricksMCPClient

client = DatabricksMCPClient(server_url, elicitation_callback=my_callback)
```

For local or CLI use, a ready-made interactive handler is provided. It prints a security warning, asks for confirmation on stdin, and opens the approved URL in a browser (running the blocking prompt off the event loop so it is safe from async code). Only URL-mode elicitation is supported; form-mode requests are rejected:

```python
from databricks_mcp import DatabricksMCPClient, interactive_url_elicitation_callback

client = DatabricksMCPClient(
    server_url, elicitation_callback=interactive_url_elicitation_callback
)
```

---

## Contribution Guide
We welcome contributions! Please see our [contribution guidelines](https://github.com/databricks/databricks-ai-bridge/tree/main/mcp) for details.

## License
This project is licensed under the [MIT License](LICENSE).

Thank you for using MCP Servers on Databricks!

