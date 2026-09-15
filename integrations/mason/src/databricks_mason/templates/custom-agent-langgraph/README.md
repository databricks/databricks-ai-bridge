# Custom LangGraph Server

This template shows how to serve a LangGraph agent with an ordinary FastAPI application. It does
not use Mason's `AgentApp` HTTP server or durable runtime.

This template does not load managed tool bindings from `agent.toml`, so `mason tools add` is not
supported. Wire framework-native Python tools and MCP servers directly in `agent/agent.py`.

```bash
mason dev
```

Call its single foreground endpoint:

```bash
curl -sS http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"input":[{"role":"user","content":"Hello"}]}'
```

Edit `runtime/main.py` to define your own HTTP contract. Edit `agent/agent.py` to change the model
or agent behavior. Deploy with `mason --profile <profile> deploy custom-agent-langgraph`.
