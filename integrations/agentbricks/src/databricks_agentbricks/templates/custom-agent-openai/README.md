# Custom OpenAI Agents server

This template shows how to serve an OpenAI Agents SDK agent with an ordinary FastAPI application.
It does not use the managed `DurableAgentServer` HTTP server or durable runtime.

This template does not load managed tool bindings from `agent.toml`, so `ab tools add` is not
supported. Wire framework-native Python tools and MCP servers directly in `agent/agent.py`.

```bash
ab dev
```

Call its single foreground endpoint:

```bash
curl -sS http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -H 'X-Databricks-Session-Id: example-session' \
  -d '{"input":[{"role":"user","content":"Hello"}]}'
```

The session header is optional here: this custom server is stateless and does not use it.
It does not provide the managed server's session queueing or persistence.

Edit `runtime/main.py` to define your own HTTP contract. Edit `agent/agent.py` to change the model
or agent behavior. Deploy with `ab --profile <profile> deploy custom-agent-openai`.
