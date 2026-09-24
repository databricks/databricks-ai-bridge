# Agent Development Guide

This project is an OpenAI Agents SDK workload hosted by `databricks_agentkit.DurableAgentServer`.

Read [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md) before changing integration points. It owns command
requirements, tool/state/tracing wiring, and recovery. [README.md](README.md) owns setup and client
examples. Keep this file as a development map rather than repeating those rules.

## Commands

```bash
ab dev
uv run pytest
ab --profile <profile> deploy <name> --source .
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
| Framework-native agent and `run_agent` | `agent/agent.py` |
| Local tools | `agent/tools/` |
| MCP servers | `agent/mcps.py` |
| Managed runtime `invoke`/`recover` hooks and input/output translation | `runtime/adapter.py` |
| `DurableAgentServer` construction and hook registration | `runtime/main.py` |
| Browser and managed-state routes | `runtime/ui.py` |
| Browser behavior | `ui/app.js` |

Keep `agent/agent.py` runnable without runtime request or context types. If you bring an existing agent,
put its framework-native execution in `run_agent`. The small `runtime/adapter.py` is the agent-author
integration point: it translates the application payload, calls `run_agent`, emits runtime events, and
shapes the response. Its `recover` hook calls the same `run_agent` with the original application input
plus a developer instruction warning that the prior attempt may have partially completed because the
Agents SDK does not expose checkpoint continuation.

## State and recovery

- Invocation state/events: in-memory in `ab dev`; Lakebase when `ab deploy` attaches a Runtime
  Store.
- Conversation transcript: in-process in `ab dev`; managed Session Store when bound, on `ab deploy`.
- Long-term memory: off in `ab dev`; managed Memory Store when bound, on `ab deploy`.
- OpenAI HITL `RunState`: process-local even with Session Store; it does not survive worker loss.
- Recovery: replay the persisted application input against the same session.

The adapter sends every translated framework event through `context.emit()` before delivery. OpenAI
Agents SDK does not expose node-level checkpoint continuation, so side effects remain at-least-once
and tools must be idempotent.

## Tools

`agent/tools/all_tools()` auto-imports tool modules. Add a decorated tool file rather than manually
editing a registry. Approval tools must declare `needs_approval=True` and appear in
`REQUIRE_APPROVAL`.
