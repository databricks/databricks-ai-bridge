# Custom LangGraph server

This template shows how to serve a LangGraph agent with an ordinary FastAPI application. It does
not use the managed `DurableAgentServer` HTTP server or durable runtime.

This template does not load managed tool bindings from `agent.toml`, so `agentbricks tools add` is not
supported. Wire framework-native Python tools and MCP servers directly in `agent/agent.py`.

Claude Chat Completions use prompt caching by default when you select a Claude model.
Set `DATABRICKS_CLAUDE_PROMPT_CACHING=off` to opt out. Other providers keep their request format. Cache hits
require a sufficiently long, unchanged prefix within the cache lifetime; cache writes carry an
additional charge.


Claude caching is configured by the AgentKit provider adapter. Sonnet 4.6's verified endpoint
names use native automatic conversation placement plus a fixed system point; older Claude
names use explicit points. Caller-supplied controls take precedence. Unknown aliases need
manual controls. The setting does not affect caching managed by other providers.

```bash
agentbricks dev
```

Call its single foreground endpoint:

```bash
curl -sS http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"input":[{"role":"user","content":"Hello"}]}'
```

Edit `runtime/main.py` to define your own HTTP contract. Edit `agent/agent.py` to change the model
or agent behavior. Deploy with `agentbricks --profile <profile> deploy custom-agent-langgraph`.
