# Agent Development Guide

This project is an OpenAI Agents SDK workload hosted by `databricks_agentkit.DurableAgentServer`.

For CLI workflows, read [.agents/skills/agent-bricks-workflow/SKILL.md](.agents/skills/agent-bricks-workflow/SKILL.md).
If it is absent, read the bundled skill at the path printed by the CLI's coding-agent hint.

Read [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md) before changing integration points. It owns command
requirements, tool/state/tracing wiring, and recovery. [README.md](README.md) owns setup and client
examples. Keep this file as a development map rather than repeating those rules.

## Commands

```bash
agentbricks dev
uv run pytest
agentbricks --profile <profile> deploy <name> --source .
```

### Deploying from a coding agent

A first deploy takes ~3 min (mostly starting app compute); redeploys are faster. Either run
`agentbricks deploy` in the foreground with a long command timeout (for example, Claude Code's Bash
`timeout: 600000`), or run `agentbricks deploy <name> --no-wait` and then
`agentbricks deployments wait agent-bricks-<name>` until it exits 0. Exit 75 means it is still
deploying; run the same command again. Don't poll with `sleep`.

## Request contract

Use only `/api/invocations`. The transport body is:

```json
{
  "id": "<uuid>",
  "session_id": "<stable-application-session>",
  "input": {
    "messages": [{"role": "user", "content": "hello"}],
    "resume": null,
    "model": "optional-serving-endpoint"
  },
  "background": false,
  "stream": false
}
```

`id` is the invocation identifier and idempotency key. `session_id`, `background`, and `stream` are
transport fields. Everything framework-specific belongs inside `input`. The browser generates a
stable session ID in local storage and sends it as top-level `session_id` on every invocation. It
also uses that value in the `X-Routing-Key` header to pin the session to one runtime replica. API
clients should do the same. That header is only for sticky routing and is not authentication or
application session state.

## Code map

| Change | File |
| --- | --- |
| Framework-native agent and `run_agent` | `agent/agent.py` |
| Harness checkpoint persistence through the Session Store | `agent/checkpoints.py` |
| Local tools | `agent/tools/` |
| MCP servers | `agent/mcps.py` |
| Managed runtime `invoke`/`recover` hooks and input/output translation | `runtime/adapter.py` |
| `DurableAgentServer` construction and hook registration | `runtime/main.py` |
| Browser and managed-state routes | `runtime/ui.py` |
| Browser behavior | `ui/app.js` |

Keep `agent/agent.py` runnable without runtime request or context types. If you bring an existing agent,
put its framework-native execution in `run_agent`. The small `runtime/adapter.py` is the agent-author
integration point: it translates the application payload, calls `run_agent`, emits runtime events, and
shapes the response. Its `recover` hook calls the same `run_agent` with the original input and a
developer instruction warning that the prior attempt may have partially completed. An already
accepted approval is never rerun; the Agents SDK has no node-level checkpoints.

## State and recovery

- Invocation state/events: in-memory in `agentbricks dev`; Lakebase when `agentbricks deploy` attaches a Runtime
  Store.
- Conversation transcript: in-process in `agentbricks dev`; managed Session Store when bound, on `agentbricks deploy`.
- Long-term memory: off in `agentbricks dev`; managed Memory Store when bound, on `agentbricks deploy`.
- OpenAI HITL `RunState`: the example harness saves it in a separate Session Store session;
  survives worker loss when a Session Store is bound, and stays process-local otherwise.
- Recovery: replay the persisted application input against the same session, except an already
  accepted approval. See [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md) for approval semantics.

The adapter sends every translated framework event through `context.emit()` before delivery. OpenAI
Agents SDK does not expose node-level checkpoint continuation, so replayed turns may repeat side
effects and tools must be idempotent.

## Tools

`agent/tools/all_tools()` auto-imports tool modules. Add a decorated tool file rather than manually
editing a registry. Approval tools must declare `needs_approval=True` and appear in
`REQUIRE_APPROVAL`. For CLI-declared MCP tools, add the exposed tool names to `REQUIRE_APPROVAL`;
the template forwards that policy to the MCP servers. Extra servers keep their own SDK policy.
