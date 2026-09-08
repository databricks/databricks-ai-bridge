# Custom OpenAI Agents Server

This template shows how to serve an OpenAI Agents SDK agent with an ordinary FastAPI application.
It does not use Mason's `AgentApp` HTTP server or durable runtime.

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
or agent behavior. Deploy with `mason --profile <profile> deploy custom-agent-openai`.
