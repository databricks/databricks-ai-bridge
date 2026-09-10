# Mason Claude Agent

An agent built on the **Anthropic Python SDK Tool Runner** (`client.beta.messages.tool_runner` /
`@beta_tool`), served by `databricks_mason.AgentApp` on Mason's durable runtime. It calls Claude
through the Anthropic API.

## Model access (required)

Unlike the other Mason templates, this one calls the Anthropic API rather than Databricks Model
Serving, so it needs an Anthropic credential:

- set **`ANTHROPIC_API_KEY`** (from the Anthropic Console), or
- set **`CLAUDE_CODE_USE_BEDROCK=1`** to route through Amazon Bedrock using the app's AWS identity.

Default model is `claude-opus-5`; override with `AGENT_CLAUDE_MODEL` or per request via `input.model`.

## Run locally

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY
mason dev
```

The API is at `http://localhost:8000/api/invocations`. Every request supplies a UUID `id`
(idempotency key); agent-specific values live in `input`:

```bash
SESSION_ID=$(uuidgen); INVOCATION_ID=$(uuidgen)
curl -sS http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"session_id\":\"$SESSION_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it? Use your tool.\"}]}}"
```

Reuse `SESSION_ID` for multi-turn history; generate a new `INVOCATION_ID` per turn.

## Human approval

`send_message` is gated (listed in `REQUIRE_APPROVAL` in `agent/agent.py`). When the model calls it
the agent stops before running it and returns an `interrupt`; resume with the same session id:

```json
{"id": "<new-uuid>", "input": {"session_id": "<same-session-id>", "resume": {"decisions": [{"type": "approve"}]}}}
```

Approval pauses are in-process (not durable), matching the other Mason templates.

## Configure and deploy

- Change the model/system prompt and tool assembly in `agent/agent.py`.
- Add local tools under `agent/tools/` (`@beta_tool`); modules are auto-discovered.
- Add MCP servers in `agent/mcps.py` or with `mason tools add mcp`.
- Bind long-term memory with `mason memory bind <store>`; durable history with `mason sessions bind <store>`.

```bash
mason --profile <profile> deploy agent-claude --source .
```

Set `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_USE_BEDROCK`) in `app.yaml`'s `env` before deploying.
