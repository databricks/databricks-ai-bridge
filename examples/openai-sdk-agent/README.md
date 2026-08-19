# Durable OpenAI Agents SDK App

This example is the PR-review agent from the custom-runtime experiment, with
its application-owned durability package replaced by `DatabricksDurableRuntime`.
The agent loop in `review_agent.py` and the OpenAI Agents SDK session in
`sessions.py` remain application concerns.

See [Live Test Observations](./OBSERVATIONS.md) for blocking, cache/conflict,
client-disconnect, and real App stop/start recovery results.

## Responsibilities

```text
client
  -> FastAPI adapter (app.py)
       -> DatabricksDurableRuntime
            -> Lakebase: openai_sdk_agent_durability.executions
            -> executor (execute_durable_review)
                 -> OpenAI Agents SDK + tools
                 -> Lakebase: openai_sdk_agent_sessions.agent_messages
```

`DatabricksDurableRuntime` owns request/response persistence, exact-request
idempotency, heartbeats, stale-attempt claims, and process-start recovery. The
executor owns the SDK session and recovery behavior. On attempt 1 it starts the
review from the request. On attempt 2 or later it reopens the same SDK session
and supplies only the fixed recovery note; it does not reconstruct an agent
prompt from the durability request.

This example intentionally allows one durable request per SDK session, so
`custom_inputs.session_id` is also the runtime `execution_id`. A multi-turn
application should use a separate execution ID for each invocation and keep its
conversation or session ID in the persisted request.

## HTTP contract

- `POST /responses` and `POST /invocations` run in blocking mode by default.
- Set `background: true` to receive `202` with an ID and poll
  `GET /responses/{execution_id}`.
- Repeating the same normalized request and ID returns the cached response.
- Reusing an ID with a different request returns `409 Conflict`.
- `background` and `stream` are transport fields and are not persisted.
  Streaming is rejected because this example does not implement it.

Clients should supply a stable `custom_inputs.session_id`. A generated ID can
be returned to a connected client, but a blocking client that loses its
connection before receiving that ID cannot later identify the execution.

Example background request:

```bash
SESSION_ID="review-$(date -u +%Y%m%dT%H%M%SZ)"

curl -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{
    \"background\": true,
    \"input\": [{\"role\": \"user\", \"content\": \"Execute the complete PR CUJ.\"}],
    \"custom_inputs\": {
      \"session_id\": \"$SESSION_ID\",
      \"pr_url\": \"https://github.com/databricks/databricks-ai-bridge/pull/459\",
      \"minimum_minutes\": 0
    }
  }"

curl "$APP_URL/responses/$SESSION_ID" \
  -H "Authorization: Bearer $APP_TOKEN"
```

## Lakebase state

Both stores use the App's `postgres` resource but separate schemas:

| Owner | Schema and table | Persisted state |
| --- | --- | --- |
| Runtime | `openai_sdk_agent_durability.executions` | execution ID, status, attempt, heartbeat, normalized request, final response |
| OpenAI Agents SDK | `openai_sdk_agent_sessions.agent_messages` | replayable user, assistant, tool-call, and tool-output items |

The runtime provides at-least-once recovery. Pod-local files and in-flight tool
processes do not survive a crash, and tools must tolerate retries.

## Deploy

Install from the repository checkout while developing this unreleased runtime:

```bash
uv venv
uv pip install -e '../..[memory]' -e '../../integrations/openai[memory]'
uv pip install 'openai-agents>=0.19.4,<0.20' 'mcp>=1.29.0,<2' \
  'mlflow>=3.10.1' 'fastapi>=0.129.0' 'uvicorn>=0.41.0'
```

For Databricks Apps, configure one Lakebase branch/database and one secret, then
deploy with an explicitly selected profile:

```bash
databricks bundle deploy -t dev --profile <PROFILE> \
  --var="lakebase_branch=projects/<project>/branches/<branch>" \
  --var="lakebase_database=projects/<project>/branches/<branch>/databases/<database>" \
  --var="openai_secret_scope=<scope>" \
  --var="openai_secret_key=<key>"

databricks bundle run open_ai_sdk_agent -t dev --profile <PROFILE> \
  --var="lakebase_branch=projects/<project>/branches/<branch>" \
  --var="lakebase_database=projects/<project>/branches/<branch>/databases/<database>" \
  --var="openai_secret_scope=<scope>" \
  --var="openai_secret_key=<key>"
```

After the runtime is released, the App build installs `requirements.txt`
directly. When deploying this PR before release, replace the
`databricks-ai-bridge` requirement with an installable wheel or Git ref that
contains `DatabricksDurableRuntime`.
