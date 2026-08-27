# Agent Development Guide

This is a LangGraph agent hosted by `DatabricksDurableApp`. The SDK owns the HTTP server,
background execution, polling, recovery, and SSE replay.

## Run and test

```bash
mason dev
uv run pytest
```

## Edit map

| Change | File |
| --- | --- |
| Model or agent behavior | `agent/agent.py` |
| Add a tool | `agent/tools/` and the tool list in `create_agent_graph()` |
| Conversation checkpointer | `agent/session_store.py` |
| App command or deployed env | `app.yaml` |
| Tests | `tests/test_agent.py` |

## Runtime contract

- The agent payload stays in the JSON request body.
- `Idempotency-Key` carries the durable run ID.
- `X-Routing-Key` carries the conversation ID.
- Missing IDs are generated and returned as response headers.
- The entrypoint receives the payload plus `DurableAgentContext`.
- Call `context.emit(event)` to persist stream events before delivery.

Do not add FastAPI routes or a background-run store to the template. Those belong in
`DatabricksDurableApp`.

## Storage

- Local default: process-local runtime state and `InMemorySaver` checkpoints.
- `AGENT_SESSION_STORE` set: Lakebase runtime state and `AsyncCheckpointSaver` checkpoints.

Use the in-memory default for local development. The managed Session Store path is intended for the
deployed app service principal.
