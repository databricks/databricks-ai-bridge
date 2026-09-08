# Agent Development Guide

This project is a LangGraph workload hosted by `databricks_mason.AgentApp`.

## Commands

```bash
mason dev
uv run pytest
mason --profile <profile> deploy <name> --source .
```

## Request contract

Use only `/api/invocations`. The transport body is:

```json
{
  "id": "<uuid>",
  "input": {
    "session_id": "<stable-application-session>",
    "messages": [{"role": "user", "content": "hello"}],
    "resume": null,
    "model": "optional-serving-endpoint"
  },
  "background": false,
  "stream": false
}
```

`id` is the invocation identifier and idempotency key. `background` and `stream` are transport
fields. Everything framework-specific belongs inside `input`. The browser generates a stable
session ID in local storage; API clients should do the same. The Apps router cookie is only for
sticky routing and is not authentication or application session state.

## Code map

| Change | File |
| --- | --- |
| Model, tools, HITL, event mapping | `agent/agent.py` |
| Local tools | `agent/tools/` |
| MCP servers | `agent/mcps.py` |
| Mason server and durable-runtime option | `runtime/main.py` |
| Browser and managed-state routes | `runtime/ui.py` |
| Browser behavior | `ui/app.js` |

Do not add another HTTP runtime. `runtime/main.py` must stay a thin layer that constructs `AgentApp`,
registers `invoke`, registers `on_recovery` when `DURABLE_RUNTIME` is enabled, and optionally
installs the UI.

## State and recovery

- Invocation state/events: in-memory in `mason dev`; Lakebase after deploy when `DURABLE_RUNTIME`
  is enabled; process-local after deploy when disabled.
- Conversation checkpoints: in-process by default; managed Session Store when bound.
- Long-term memory: managed Memory Store when bound.
- LangGraph HITL: checkpointed with the conversation and durable when Session Store is bound.
- Recovery: continue a checkpoint tagged with the current invocation ID; otherwise replay input.

Every framework event must pass through `context.emit()` before delivery. Checkpoints use
`durability="sync"` so acknowledged progress is available to a replacement worker. External side
effects remain at-least-once; tools must be idempotent.

## Tools

`agent/tools/all_tools()` auto-imports tool modules. Add a decorated tool file rather than manually
editing a registry. Add tools requiring approval to `REQUIRE_APPROVAL`.
