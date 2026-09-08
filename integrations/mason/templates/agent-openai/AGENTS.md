# Agent Development Guide

This project is an OpenAI Agents SDK workload hosted by `databricks_mason.DurableAgentApp`.

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
| Durable app wiring | `runtime/main.py` |
| Browser and managed-state routes | `runtime/ui.py` |
| Browser behavior | `ui/app.js` |

Do not add another HTTP runtime. `runtime/main.py` must stay a thin layer that constructs
`DurableAgentApp`, registers `invoke`, conditionally registers `on_recovery` when automatic recovery
is enabled, and optionally installs the UI.

## State and recovery

- Invocation state/events: in-memory in `mason dev`, Lakebase after `mason deploy`.
- Conversation transcript: in-process by default; managed Session Store when bound.
- Long-term memory: managed Memory Store when bound.
- OpenAI HITL `RunState`: process-local even with Session Store; it does not survive worker loss.
- Recovery: replay the persisted application input against the same session.

Every framework event must pass through `context.emit()` before delivery. OpenAI Agents SDK does
not expose node-level checkpoint continuation, so side effects remain at-least-once and tools must
be idempotent.

## Tools

`agent/tools/all_tools()` auto-imports tool modules. Add a decorated tool file rather than manually
editing a registry. Approval tools must declare `needs_approval=True` and appear in
`REQUIRE_APPROVAL`.
