# Agent Development Guide

This project is an Anthropic Tool Runner workload hosted by `databricks_mason.AgentApp`.

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
    "model": "optional-claude-model"
  },
  "background": false,
  "stream": false
}
```

`id` is the invocation identifier and idempotency key. Everything framework-specific belongs in
`input`. The Apps router cookie is only for sticky routing, not application session state.

## Model access

Claude is reached through the Anthropic API — set `ANTHROPIC_API_KEY`, or `CLAUDE_CODE_USE_BEDROCK=1`
for Bedrock. Default model `claude-opus-5` (`AGENT_CLAUDE_MODEL` / `input.model` override).

## Code map

| Change | File |
| --- | --- |
| Model, tools, HITL, event mapping | `agent/agent.py` |
| Local tools (`@beta_tool`) | `agent/tools/` |
| MCP servers | `agent/mcps.py` |
| Mason server and durable-runtime option | `runtime/main.py` |

`runtime/main.py` must stay a thin layer: construct `AgentApp`, register `invoke`, and register
`on_recovery` when `DURABLE_RUNTIME` is enabled.

## State and recovery

- Invocation state/events: in-memory in `mason dev`; Lakebase after deploy (durable runtime on).
- Conversation history: in-process by default; managed Session Store when bound (`databricks_mason.claude.session_history`).
- Long-term memory: managed Memory Store when bound (`memory_tools`).
- Human approval: gated tools in `REQUIRE_APPROVAL` surface an `interrupt`; pauses are in-process.
- Recovery: the Tool Runner owns its loop, so recovery replays input against the persisted transcript.

## Tools

`agent/tools/all_tools()` auto-imports tool modules. Add a `@beta_tool`-decorated function in a new
file rather than editing a registry. Gate an action tool by adding its name to `REQUIRE_APPROVAL`.
