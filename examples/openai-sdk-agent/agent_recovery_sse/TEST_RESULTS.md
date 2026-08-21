# Agent Recovery With SSE Replay: Test Results

Captured on 2026-08-20 and 2026-08-21 against:

- App: `openai-agent-session-sse`
- Active deployment: `01f19d00fc55185289e487ade308ebf8`
- Lakebase branch: `durable-session-sse`
- Runtime schema: `agent_server`
- SDK schema: `openai_agent_sse_sessions`
- PR under review: `https://github.com/databricks/databricks-ai-bridge/pull/459`

## Setup

```bash
PROFILE=ml-inference-staging
APP_URL='https://openai-agent-session-sse-1653573648247579.staging.aws.databricksapps.com'
APP_TOKEN=$(env -u DATABRICKS_HOST -u DATABRICKS_TOKEN \
  -u DATABRICKS_CONFIG_PROFILE \
  databricks auth token --profile "$PROFILE" -o json | jq -r '.access_token')

ENDPOINT='projects/shivam-openai-agent-on-apps/branches/durable-session-sse/endpoints/primary'
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

The initial request input was:

```json
[{"role":"user","content":"Execute the complete PR CUJ."}]
```

The normal handler parsed that text and called the shared review loop. The
normalized SDK prompt contained the PR URL, temporary workspace, four CUJs,
and the client text.

For agent-managed recovery, `@on_resume()` replaces `request.input` with:

```text
[RECOVERY] The previous attempt was interrupted. Continue the task using the
transcript already persisted by the agent's session store. Inspect external
side effects and safely repeat any interrupted operation.
```

The same `@stream()` handler receives the translated request. It does not
inspect a recovery marker or select a separate implementation. It reopens the
same SDK session, whose persisted transcript supplies the earlier messages.

## Test 1: Generated Session Anchor Happy Path

This request intentionally omitted all session-anchor fields:

```bash
jq -nc '{
  background:true,
  input:[{role:"user",content:"Execute the complete PR CUJ."}],
  custom_inputs:{
    pr_url:"https://github.com/databricks/databricks-ai-bridge/pull/459",
    minimum_minutes:0
  }
}' >/tmp/agent-sse-happy-request.json

curl -sS -o /tmp/agent-sse-happy-start.json -w 'http=%{http_code}\n' \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @/tmp/agent-sse-happy-request.json
```

Client observation:

```text
http=200
{"id":"resp_c7a860f3a4b34e68ba1eb749","status":"in_progress","error":null}
```

Polling later returned `status=completed` and one output item. Before the first
handler call, the server warned and persisted
`context.conversation_id=resp_c7a860f3a4b34e68ba1eb749`.

Final Lakebase state:

| Field | Value |
| --- | --- |
| status / attempt / streaming | `completed / 1 / false` |
| request / response size | `550 / 2,053` characters |
| generated SDK session anchor | `resp_c7a860f3a4b34e68ba1eb749` |
| runtime event rows | `1`, sequence `0` |
| SDK messages | `24`, IDs `199-246` |
| terminal output items | `1` |

## Test 2: Client Disconnect And Cursor Replay

The completed streaming run used session
`test-02-disconnect-agent-sse-20260820T192229Z`:

```bash
jq -nc '{
  background:true,
  stream:true,
  input:[{role:"user",content:"Execute the complete PR CUJ."}],
  custom_inputs:{
    session_id:"test-02-disconnect-agent-sse-20260820T192229Z",
    pr_url:"https://github.com/databricks/databricks-ai-bridge/pull/459",
    minimum_minutes:0
  }
}' >/tmp/agent-sse-disconnect-request.json

curl -sS -N --max-time 8 -o /tmp/agent-sse-disconnect-first.sse \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @/tmp/agent-sse-disconnect-request.json
```

The client extracts the durable response ID and last sequence from any received
SSE JSON object, then reconnects:

```bash
RESPONSE_ID=$(rg '^data: ' /tmp/agent-sse-disconnect-first.sse \
  | sed 's/^data: //' | rg -v '^\[DONE\]$' \
  | jq -r 'select(.response_id != null) | .response_id' | head -1)
LAST_SEQUENCE=$(rg '^data: ' /tmp/agent-sse-disconnect-first.sse \
  | sed 's/^data: //' | rg -v '^\[DONE\]$' \
  | jq -s 'map(.sequence_number) | max')

curl -sS -N \
  "$APP_URL/responses/$RESPONSE_ID?stream=true&starting_after=$LAST_SEQUENCE" \
  -H "Authorization: Bearer $APP_TOKEN"
```

Captured completed run:

| Field | Value |
| --- | --- |
| response ID | `resp_d55a661e2da746ffaf4ac240` |
| status / attempt / streaming | `completed / 1 / true` |
| request / response size | `565 / 49,620` characters |
| durable event log | `1,660` rows, sequences `0-1659` |
| SDK session | `test-02-disconnect-agent-sse-20260820T192229Z` |
| SDK messages | `30`, IDs `21-50` |
| terminal output items | `29` |

Replay was rerun on 2026-08-21 with `starting_after=10`:

- HTTP `200`.
- `1,649` events were returned, sequences `11-1659`.
- Every event retained top-level response ID
  `resp_d55a661e2da746ffaf4ac240`.
- The last event was `response.completed`, followed by one `[DONE]` frame.

## Test 3: Disconnect, Pod Crash, Recovery, And Repeated Replay

This test ran against the pushed recovery-agnostic handlers on 2026-08-21. It
omitted a client session ID so the server-generated anchor could also be
verified.

Start the stream and force the first client disconnect after eight seconds:

```bash
jq -nc '{
  background:true,
  stream:true,
  input:[{role:"user",content:"Execute the complete PR CUJ."}],
  custom_inputs:{
    pr_url:"https://github.com/databricks/databricks-ai-bridge/pull/459",
    minimum_minutes:0
  }
}' >/tmp/agent-sse-current-request.json

curl -sS -N --max-time 8 -o /tmp/agent-sse-current-initial.sse \
  -w 'http=%{http_code}\n' \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @/tmp/agent-sse-current-request.json
```

First client observation:

| Field | Value |
| --- | --- |
| curl result | exit `28` after 8 seconds, HTTP `200` |
| durable response ID | `resp_6db34436d1f5425985bc8983` |
| received events | `245` |
| received sequence range | `0-244` |
| `[DONE]` | not received; task remained active |

Immediately before stopping the App, Lakebase had advanced beyond the client:

| Store | Pre-crash state |
| --- | --- |
| Runtime response | `in_progress`, attempt `1`, streaming `true` |
| Generated conversation/session anchor | `resp_6db34436d1f5425985bc8983` |
| Durable events | `391`, sequences `0-390` |
| SDK transcript | `7` messages, IDs `260-266` |

The App was stopped and restarted:

```bash
env -u DATABRICKS_HOST -u DATABRICKS_TOKEN -u DATABRICKS_CONFIG_PROFILE \
  databricks apps stop openai-agent-session-sse --profile "$PROFILE"
env -u DATABRICKS_HOST -u DATABRICKS_TOKEN -u DATABRICKS_CONFIG_PROFILE \
  databricks apps start openai-agent-session-sse --profile "$PROFILE"
```

The restarted server found the stale attempt-1 heartbeat and atomically
advanced the response to attempt `2`. The client then reconnected from the
sequence it had actually received, not from the larger database sequence:

```bash
curl -sS -N --max-time 600 \
  "$APP_URL/responses/resp_6db34436d1f5425985bc8983?stream=true&starting_after=244" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -o /tmp/agent-sse-current-replay.sse
```

First replay observation:

| Field | Value |
| --- | --- |
| HTTP | `200` |
| returned events | `509` |
| sequence range | `245-753` |
| recovery sentinel | one `response.resumed`, attempt `2`, sequence `506` |
| sentinel conversation ID | `resp_6db34436d1f5425985bc8983` |
| final frame on this connection | raw model `response.completed`, sequence `753` |
| `[DONE]` | not received; authoritative response was still `in_progress` |

The transport closed while the agent remained healthy. Polling showed attempt
`2` with a fresh heartbeat. A second reconnect from sequence `753` waited five
minutes and timed out with HTTP `200` but zero bytes because no new stream event
was available:

```bash
curl -sS -N --max-time 300 \
  "$APP_URL/responses/resp_6db34436d1f5425985bc8983?stream=true&starting_after=753" \
  -H "Authorization: Bearer $APP_TOKEN"
```

This is an important client contract: a live agent may produce no SSE bytes
while a tool runs. Clients should retain the response ID and cursor, reconnect
after transport/idle timeouts, and use non-streaming `GET /responses/{id}` when
they need authoritative status.

### Final snapshot of this recovered attempt

The recovered attempt did not reach a terminal state during the bounded test.
After more than 20 minutes, the authoritative client request still returned:

```bash
curl -sS \
  "$APP_URL/responses/resp_6db34436d1f5425985bc8983" \
  -H "Authorization: Bearer $APP_TOKEN" | jq \
  '{id,status,error,output_items:(.output|length)}'
```

```json
{
  "id": "resp_6db34436d1f5425985bc8983",
  "status": "in_progress",
  "error": null,
  "output_items": 0
}
```

A final reconnect from the latest cursor also received no new event:

```bash
curl -sS -N --max-time 10 \
  "$APP_URL/responses/resp_6db34436d1f5425985bc8983?stream=true&starting_after=753" \
  -H "Authorization: Bearer $APP_TOKEN"
```

```text
curl: (28) Operation timed out after 10001 milliseconds with 0 bytes received
http=200
```

The final Lakebase snapshot at `2026-08-21T02:23:47Z` was:

| Store | State |
| --- | --- |
| Runtime response | `in_progress`, attempt `2`, streaming `true` |
| Runtime request / response | `550` characters / `NULL` |
| Heartbeat | `2026-08-21 02:23:57.040085+00`, about two seconds old when queried |
| Attempt-1 events | `506` rows, sequences `0-505` |
| Attempt-2 events | `248` rows, sequences `506-753` |
| SDK transcript | Same generated session, `12` messages, IDs `260-271` |
| Initial SDK user input | ID `260`, `936` characters, no recovery marker |
| Recovery SDK user input | ID `269`, `1,118` characters, `[RECOVERY]` at character `908` |

The replay from cursor `244` therefore returned all `261` missing attempt-1
events (`245-505`) and all `248` attempt-2 events (`506-753`). This verifies
durable cursor replay across the pod restart and verifies that recovery reopened
the same SDK transcript with the translated recovery input. It does **not**
verify terminal completion for this particular run: no terminal runtime
response or `[DONE]` frame was persisted, even though the server heartbeat
remained fresh.

## SQL Used To Verify State

```sql
SELECT response_id,status,attempt_number,is_streaming,heartbeat_at,
       now()-heartbeat_at AS heartbeat_age,
       original_request::jsonb #>> '{context,conversation_id}' AS conversation_id,
       length(original_request) AS request_chars,
       length(response) AS response_chars,
       CASE WHEN response IS NULL THEN NULL
            ELSE jsonb_array_length(response::jsonb->'output') END AS output_items
FROM agent_server.responses
WHERE response_id = 'resp_6db34436d1f5425985bc8983';

SELECT attempt_number,count(*) AS rows,
       min(sequence_number) AS min_seq,max(sequence_number) AS max_seq
FROM agent_server.messages
WHERE response_id = 'resp_6db34436d1f5425985bc8983'
GROUP BY attempt_number
ORDER BY attempt_number;

SELECT session_id,count(*) AS messages,min(id),max(id),
       min(created_at) AS first_at,max(created_at) AS last_at
FROM openai_agent_sse_sessions.agent_messages
WHERE session_id = 'resp_6db34436d1f5425985bc8983'
GROUP BY session_id;

SELECT id,length(message_data::jsonb->>'content') AS content_chars,
       strpos(message_data::jsonb->>'content','[RECOVERY]') AS recovery_offset,
       left(replace(message_data::jsonb->>'content',E'\n',' '),180) AS content_prefix,
       created_at
FROM openai_agent_sse_sessions.agent_messages
WHERE session_id = 'resp_6db34436d1f5425985bc8983'
  AND message_data::jsonb->>'role' = 'user'
ORDER BY id;
```
