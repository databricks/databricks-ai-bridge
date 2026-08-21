# Framework Recovery With SSE Replay: Test Results

Captured on 2026-08-20 and re-queried/replayed on 2026-08-21 against:

- App: `openai-agent-framework-sse`
- Active deployment: `01f19ce6d5d11d82a51f56e1f993eb18`
- Lakebase branch: `durable-framework-sse`
- Runtime schema: `agent_server`
- SDK schema: `openai_framework_sse_sessions`
- PR under review: `https://github.com/databricks/databricks-ai-bridge/pull/459`

## Setup

```bash
PROFILE=ml-inference-staging
APP_URL='https://openai-agent-framework-sse-1653573648247579.staging.aws.databricksapps.com'
APP_TOKEN=$(env -u DATABRICKS_HOST -u DATABRICKS_TOKEN \
  -u DATABRICKS_CONFIG_PROFILE \
  databricks auth token --profile "$PROFILE" -o json | jq -r '.access_token')
```

Lakebase was inspected with:

```bash
ENDPOINT='projects/shivam-openai-agent-on-apps/branches/durable-framework-sse/endpoints/primary'
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

The request input was:

```json
[{"role":"user","content":"Execute the complete PR CUJ."}]
```

The handler parses the latest request input and passes it to the shared review
agent. The normalized first-attempt prompt persisted by the SDK was 908
characters and had this shape:

```text
PR: https://github.com/databricks/databricks-ai-bridge/pull/459
Iteration: 1
Workspace: <temporary-workspace>

First clone the repository into <temporary-workspace>/repo and check out the PR head...
1. quality: discover and run the repository's formatting, lint, and type checks
2. package-tests: run all PR-relevant package test suites and report each separately
3. build-install: build distributions, install them cleanly, and verify imports
4. agent-e2e: launch an affected example agent, invoke it, and retain useful logs
```

On recovery, `@on_resume()` runs before the normal handler. The built-in
framework policy appends prose containing the prior attempt's JSON event log
and rotates the session ID. For the captured crash, the attempt-2 SDK prompt
was 12,176 characters; `[RECOVERY]` began at character 908 and serialized
attempt-1 events with sequences `0-10`.

## Test 1: Background Invoke Happy Path

Request:

```bash
SESSION_ID='test-01-happy-framework-final-20260820T202134Z'
jq -nc --arg sid "$SESSION_ID" '{
  background:true,
  input:[{role:"user",content:"Execute the complete PR CUJ."}],
  custom_inputs:{
    session_id:$sid,
    pr_url:"https://github.com/databricks/databricks-ai-bridge/pull/459",
    minimum_minutes:0
  }
}' >/tmp/framework-happy-request.json

curl -sS -o /tmp/framework-happy-start.json -w 'http=%{http_code}\n' \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @/tmp/framework-happy-request.json
```

Client observation:

- HTTP `200`; the start response returned durable ID
  `resp_ac1d45327d4e42e0b00628d8` with `status=in_progress`.
- Polling `GET /responses/resp_ac1d45327d4e42e0b00628d8` eventually returned
  `status=completed` and one terminal output item.

Final Lakebase state:

| Field | Value |
| --- | --- |
| status / attempt / streaming | `completed / 1 / false` |
| request / response size | `566 / 2,021` characters |
| runtime event rows | `1`, sequence `0` |
| SDK session | `test-01-happy-framework-final-20260820T202134Z` |
| SDK messages | `30`, IDs `98-127` |

## Test 2: Client Disconnect And SSE Replay

Start a background stream and intentionally disconnect the client:

```bash
SESSION_ID='test-02-disconnect-framework-final-20260820T202841Z'
jq -nc --arg sid "$SESSION_ID" '{
  background:true,
  stream:true,
  input:[{role:"user",content:"Execute the complete PR CUJ."}],
  custom_inputs:{
    session_id:$sid,
    pr_url:"https://github.com/databricks/databricks-ai-bridge/pull/459",
    minimum_minutes:0
  }
}' >/tmp/framework-sse-request.json

curl -sS -N --max-time 8 -o /tmp/framework-sse-first.sse \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @/tmp/framework-sse-request.json
```

Every SSE JSON object includes top-level `response_id` and `sequence_number`.
The client records both before reconnecting:

```bash
RESPONSE_ID=$(rg '^data: ' /tmp/framework-sse-first.sse \
  | sed 's/^data: //' | rg -v '^\[DONE\]$' \
  | jq -r 'select(.response_id != null) | .response_id' | head -1)
LAST_SEQUENCE=$(rg '^data: ' /tmp/framework-sse-first.sse \
  | sed 's/^data: //' | rg -v '^\[DONE\]$' \
  | jq -s 'map(.sequence_number) | max')

curl -sS -N \
  "$APP_URL/responses/$RESPONSE_ID?stream=true&starting_after=$LAST_SEQUENCE" \
  -H "Authorization: Bearer $APP_TOKEN"
```

Captured completed run:

| Field | Value |
| --- | --- |
| response ID | `resp_b52e5721ee3b470fbac5ffcd` |
| status / attempt / streaming | `completed / 1 / true` |
| request / response size | `571 / 84,613` characters |
| terminal output items | `25` |
| durable event log | `2,608` rows, sequences `0-2607` |
| SDK session | `test-02-disconnect-framework-final-20260820T202841Z` |
| SDK messages | `26`, IDs `128-153` |

The original client can disappear without cancelling the task. A later client
uses the durable response ID plus its last sequence to replay only missing
events and then tail new events.

## Test 3: Pod Crash, Framework Recovery, And Replay

The captured crash run used session
`test-03-crash-framework-final-20260820T203044Z` and response
`resp_5bd6b74ead344cadb4aad5dc`.

The App was stopped while the response was active and then restarted:

```bash
env -u DATABRICKS_HOST -u DATABRICKS_TOKEN -u DATABRICKS_CONFIG_PROFILE \
  databricks apps stop openai-agent-framework-sse --profile "$PROFILE"
env -u DATABRICKS_HOST -u DATABRICKS_TOKEN -u DATABRICKS_CONFIG_PROFILE \
  databricks apps start openai-agent-framework-sse --profile "$PROFILE"
```

After the heartbeat became stale, the restarted server atomically claimed the
row, incremented the attempt, called `@on_resume()`, and opened the rotated SDK
session `test-03-crash-framework-final-20260820T203044Z::attempt-2`.

The replay command was rerun on 2026-08-21 from the last attempt-1 sequence:

```bash
RESPONSE_ID='resp_5bd6b74ead344cadb4aad5dc'
curl -sS -N \
  "$APP_URL/responses/$RESPONSE_ID?stream=true&starting_after=10" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -o /tmp/framework-crash-replay.sse
```

Client observation:

- HTTP `200`.
- Replayed `1,653` events, sequences `11-1663`.
- The first replayed event was `response.resumed`, attempt `2`, sequence `11`.
- Its `conversation_id` was
  `test-03-crash-framework-final-20260820T203044Z::attempt-2`.
- The last event was `response.completed`, followed by one `[DONE]` frame.

Final Lakebase state:

| Store | State |
| --- | --- |
| Runtime response | `completed`, attempt `2`, streaming `true` |
| Serialized request / response | `566 / 58,214` characters |
| Attempt-1 events | `11` rows, sequences `0-10` |
| Attempt-2 events | `1,653` rows, sequences `11-1663` |
| Initial SDK session | `1` message, ID `154` |
| Rotated attempt-2 SDK session | `22` messages, IDs `155-176` |
| Terminal output | `21` items |

This mode does not attempt to repair the crashed SDK session. Recovery input is
derived from the runtime event log and processed in a fresh, rotated session.

## SQL Used To Verify State

```sql
SELECT response_id,status,attempt_number,is_streaming,heartbeat_at,
       length(original_request) AS request_chars,
       length(response) AS response_chars,
       jsonb_array_length(response::jsonb->'output') AS output_items
FROM agent_server.responses
WHERE response_id = 'resp_5bd6b74ead344cadb4aad5dc';

SELECT response_id,attempt_number,count(*) AS rows,
       min(sequence_number) AS min_seq,max(sequence_number) AS max_seq
FROM agent_server.messages
WHERE response_id = 'resp_5bd6b74ead344cadb4aad5dc'
GROUP BY response_id,attempt_number
ORDER BY attempt_number;

SELECT s.session_id,count(m.id) AS messages,min(m.id),max(m.id)
FROM openai_framework_sse_sessions.agent_sessions s
LEFT JOIN openai_framework_sse_sessions.agent_messages m USING(session_id)
WHERE s.session_id LIKE 'test-03-crash-framework-final-20260820T203044Z%'
GROUP BY s.session_id
ORDER BY s.session_id;
```
