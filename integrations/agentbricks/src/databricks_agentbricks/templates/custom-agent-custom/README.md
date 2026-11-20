# Your framework with a custom server

This model-free echo example runs your framework or harness behind an ordinary FastAPI server.
Replace `run_agent` in `agent/agent.py` and add your framework's dependencies to `pyproject.toml`.
Edit `runtime/main.py` to choose your own HTTP contract.

```bash
agentbricks dev
curl -sS http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"input":[{"role":"user","content":"Hello"}]}'
uv run pytest
agentbricks --profile <profile> deploy <name> --source .
```

The sample returns `{"output":[{"role":"assistant","content":"Echo: Hello"}],"status":"completed"}`.
It provides one foreground endpoint without managed invocation persistence, recovery, session
history, memory integration, tool bindings, or a browser UI. For the sample that integrates managed
stores and durable invocations, create a project with `agentbricks init --framework custom` and
the default managed server. You can also call public `databricks_agentkit.AgentKitClient` store
APIs from your own server; the application must supply its identity and durability contracts.
