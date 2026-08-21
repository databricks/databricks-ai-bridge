# Agent Recovery With Polling: Test Results

Captured on 2026-08-20 and re-queried on 2026-08-21 against:

- App: `openai-agent-session-poll`
- Active deployment: `01f19ce636a61d6dad32d8e069bdabf2`
- Lakebase branch: `durable-session-poll`
- Runtime schema: `agent_server`
- SDK schema: `openai_agent_polling_sessions`
- PR under review: `https://github.com/databricks/databricks-ai-bridge/pull/459`

## Setup

```bash
PROFILE=ml-inference-staging
APP_URL='https://openai-agent-session-poll-1653573648247579.staging.aws.databricksapps.com'
APP_TOKEN=$(env -u DATABRICKS_HOST -u DATABRICKS_TOKEN \
  -u DATABRICKS_CONFIG_PROFILE \
  databricks auth token --profile "$PROFILE" -o json | jq -r '.access_token')

ENDPOINT='projects/shivam-openai-agent-on-apps/branches/durable-session-poll/endpoints/primary'
PGHOST=$(env -u DATABRICKS_HOST -u DATABRICKS_TOKEN \
  -u DATABRICKS_CONFIG_PROFILE \
  databricks postgres get-endpoint "$ENDPOINT" --profile "$PROFILE" -o json \
  | jq -r '.status.hosts.host')
PGPASSWORD=$(env -u DATABRICKS_HOST -u DATABRICKS_TOKEN \
  -u DATABRICKS_CONFIG_PROFILE \
  databricks postgres generate-database-credential "$ENDPOINT" \
  --profile "$PROFILE" -o json | jq -r '.token')
export PGPASSWORD
PGUSER=$(env -u DATABRICKS_HOST -u DATABRICKS_TOKEN \
  -u DATABRICKS_CONFIG_PROFILE \
  databricks current-user me --profile "$PROFILE" -o json | jq -r '.userName')
PG_CONNECTION="host=$PGHOST port=5432 dbname=databricks_postgres user=$PGUSER sslmode=require"
```

## Input Passed To The Agent Loop

The happy-path request used:

```json
[{"role":"user","content":"Execute the complete PR CUJ."}]
```

The normalized initial SDK prompt for the crash test was 936 characters. It
contained the PR URL, temporary workspace, four CUJs, and ended with the client
text `Execute the complete PR CUJ.`.

On attempt `2`, `@on_resume()` replaced `request.input` with the fixed recovery
message before invoking the same normal handler:

```text
[RECOVERY] The previous attempt was interrupted. Continue the task using the
transcript already persisted by the agent's session store. Inspect external
side effects and safely repeat any interrupted operation.
```

The normal handler did not detect recovery. It parsed that request prompt and
called the same shared review loop. The second SDK user entry was 1,118
characters, with `[RECOVERY]` beginning at character 908. Both entries were in
the same SDK session.

## Test 1: Background Polling Happy Path

```bash
SESSION_ID='test-01-happy-session-poll-20260820T190714Z'
jq -nc --arg sid "$SESSION_ID" '{
  background:true,
  input:[{role:"user",content:"Execute the complete PR CUJ."}],
  custom_inputs:{
    session_id:$sid,
    pr_url:"https://github.com/databricks/databricks-ai-bridge/pull/459",
    minimum_minutes:0
  }
}' >/tmp/polling-happy-request.json

curl -sS -o /tmp/polling-happy-start.json -w 'http=%{http_code}\n' \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @/tmp/polling-happy-request.json

RESPONSE_ID=$(jq -r '.id' /tmp/polling-happy-start.json)
curl -sS "$APP_URL/responses/$RESPONSE_ID" \
  -H "Authorization: Bearer $APP_TOKEN" | jq .
```

Client observation:

- HTTP `200`; start returned `resp_208374e2b2c64101af5ff3a4` with
  `status=in_progress`.
- Polling eventually returned `status=completed` and one output item.

Final Lakebase state:

| Field | Value |
| --- | --- |
| status / attempt / streaming | `completed / 1 / false` |
| request / response size | `563 / 1,910` characters |
| runtime event rows | `0` |
| SDK session | `test-01-happy-session-poll-20260820T190714Z` |
| SDK messages | `24`, IDs `1-24` |

The terminal payload comes from `agent_server.responses.response`; polling does
not reconstruct it from stream events.

## Test 2: Streaming Is Explicitly Rejected

The following commands were rerun on 2026-08-21:

```bash
jq -nc '{
  background:true,
  stream:true,
  input:[{role:"user",content:"Execute the complete PR CUJ."}],
  custom_inputs:{
    session_id:"test-stream-rejected-doc",
    pr_url:"https://github.com/databricks/databricks-ai-bridge/pull/459",
    minimum_minutes:0
  }
}' >/tmp/polling-stream-request.json

curl -sS -o /tmp/polling-stream-post.json -w 'http=%{http_code}\n' \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @/tmp/polling-stream-request.json

curl -sS -o /tmp/polling-stream-get.json -w 'http=%{http_code}\n' \
  "$APP_URL/responses/resp_208374e2b2c64101af5ff3a4?stream=true" \
  -H "Authorization: Bearer $APP_TOKEN"
```

Observed client responses:

```text
POST http=400
{"detail":"Background streaming requires SSE replay to be enabled."}

GET http=400
{"detail":"SSE replay is disabled for this server."}
```

No durability row is created for the rejected POST.

## Test 3: Generated Session Anchor And Pod Crash Recovery

This request intentionally omitted `session_id`, `thread_id`, and
`conversation_id`:

```bash
jq -nc '{
  background:true,
  input:[{role:"user",content:"Execute the complete PR CUJ."}],
  custom_inputs:{
    pr_url:"https://github.com/databricks/databricks-ai-bridge/pull/459",
    minimum_minutes:0
  }
}' >/tmp/polling-crash-request.json

curl -sS -o /tmp/polling-crash-start.json -w 'http=%{http_code}\n' \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @/tmp/polling-crash-request.json
```

Client observation before the crash:

```text
http=200
{"id":"resp_9b56b812417c4dd3a4b3939b","status":"in_progress","error":null}
```

The server logged a warning and persisted the generated anchor as
`original_request.context.conversation_id = resp_9b56b812417c4dd3a4b3939b`.
The runtime `response_id` and SDK session are different concepts; they share a
value only because this request used the generated fallback.

The App was stopped while attempt `1` was active and then restarted:

```bash
env -u DATABRICKS_HOST -u DATABRICKS_TOKEN -u DATABRICKS_CONFIG_PROFILE \
  databricks apps stop openai-agent-session-poll --profile "$PROFILE"
env -u DATABRICKS_HOST -u DATABRICKS_TOKEN -u DATABRICKS_CONFIG_PROFILE \
  databricks apps start openai-agent-session-poll --profile "$PROFILE"
```

The original HTTP request was already complete because this is background
mode. The client retained the durable response ID and polled the same URL after
restart:

```bash
curl -sS \
  "$APP_URL/responses/resp_9b56b812417c4dd3a4b3939b" \
  -H "Authorization: Bearer $APP_TOKEN" | jq .
```

Observed recovery:

- The stale row was claimed and advanced from attempt `1` to attempt `2`.
- `context.conversation_id` remained
  `resp_9b56b812417c4dd3a4b3939b`.
- The same SDK session received the initial user entry at message ID `224` and
  the recovery entry at message ID `231`.
- The response eventually returned `status=completed` with one output item.
- The SDK session finished with `27` messages, IDs `224-280`.
- `agent_server.messages` remained empty throughout recovery.

Final Lakebase state:

| Field | Value |
| --- | --- |
| status / attempt / streaming | `completed / 2 / false` |
| request / response size | `550 / 2,174` characters |
| generated conversation/session anchor | `resp_9b56b812417c4dd3a4b3939b` |
| runtime event rows | `0` |
| SDK messages | `27`, IDs `224-280` |
| terminal output items | `1` |

## SQL Used To Verify State

```sql
SELECT response_id,status,attempt_number,is_streaming,heartbeat_at,
       original_request::jsonb #>> '{context,conversation_id}' AS conversation_id,
       length(original_request) AS request_chars,
       length(response) AS response_chars
FROM agent_server.responses
WHERE response_id = 'resp_9b56b812417c4dd3a4b3939b';

SELECT count(*)
FROM agent_server.messages
WHERE response_id = 'resp_9b56b812417c4dd3a4b3939b';

SELECT id,message_data::jsonb->>'role' AS role,
       strpos(message_data::jsonb->>'content','[RECOVERY]') AS recovery_offset
FROM openai_agent_polling_sessions.agent_messages
WHERE session_id = 'resp_9b56b812417c4dd3a4b3939b'
ORDER BY id;
```
