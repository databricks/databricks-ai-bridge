# Live Test Observations

These live results were recorded for the transport-neutral runtime version of
the example before the AgentServer-derived transport was added. The inherited
server keeps the same runtime, executor recovery behavior, HTTP contract, and
Lakebase schemas, but this PR was not separately deployed for these observations.

The baseline tests ran on 2026-08-19 using an isolated App name and schemas so
they did not affect the earlier experiment.

- App: `open-ai-sdk-runtime`
- URL: `https://open-ai-sdk-runtime-1653573648247579.staging.aws.databricksapps.com`
- Lakebase branch: `projects/shivam-openai-agent-on-apps/branches/agent-app`
- Database: `databricks_postgres`
- Runtime schema: `openai_sdk_runtime_durability`
- SDK session schema: `openai_sdk_runtime_sessions`

The PR packages were built as local wheels and included in the test deployment
because `DatabricksDurableRuntime` was not released yet. All remaining packages
installed through the workspace package repository, and the App finished in
`ACTIVE / RUNNING`; no package-proxy failure occurred.

## Common request and database query

Each test supplied a stable ID in `custom_inputs.session_id`:

```bash
curl -sS --max-time 900 \
  -o response.json \
  -w 'http_code=%{http_code}\ntime_total=%{time_total}\n' \
  -X POST "$APP_URL/responses" \
  -H "Authorization: Bearer $APP_TOKEN" \
  -H 'Content-Type: application/json' \
  --data @request.json
```

The following query joined runtime state with SDK history:

```sql
SELECT e.status, e.attempt, e.heartbeat_at,
       e.request, e.response, count(m.id) AS sdk_messages
FROM openai_sdk_runtime_durability.executions e
LEFT JOIN openai_sdk_runtime_sessions.agent_messages m
  ON m.session_id = e.execution_id
WHERE e.execution_id = :execution_id
GROUP BY e.execution_id;
```

## Test 1: blocking happy path

Execution: `runtime-happy-20260819T221342Z`

```text
http_code=200
time_total=74.347104
curl_exit=0
```

Lakebase after completion:

```text
status=COMPLETED  attempt=1  sdk_messages=30
request=persisted  response=persisted
```

The runtime row contained the normalized Responses request and final response.
The SDK table independently contained the model and tool history.

## Test 2: cache and conflict

Posting the exact Test 1 request and ID again returned in 0.258 seconds. The
original and cached response files had the same SHA-256:

```text
726b28b5a378e0b6e176a9894deeceb07f28ed8c7865ba21b81fd3324a98a23e
```

Changing only the input while retaining the ID returned:

```text
http_code=409
execution 'runtime-happy-20260819T221342Z' was already accepted with a different request
```

This verifies exact-request idempotency rather than ID-only response reuse.

## Test 3: blocking client disconnect

Execution: `runtime-disconnect-20260819T221458Z`

The client process was terminated after Lakebase showed `ACTIVE`, attempt `1`,
and one SDK message:

```text
curl_exit=143
```

The runtime task continued without the HTTP client. Its final state was:

```text
status=COMPLETED  attempt=1  sdk_messages=36
request=persisted  response=persisted
```

`GET /responses/runtime-disconnect-20260819T221458Z` then returned the completed
response with HTTP `200`. Unlike the earlier custom supervisor experiment, a
disconnected client can retrieve the persisted result by its stable ID.

## Test 4: background stop/start recovery

Execution: `runtime-crash-20260819T222154Z`

The background request returned immediately:

```text
http_code=202
time_total=0.223404
status=in_progress
```

The App was stopped after the first attempt had persisted SDK history:

```text
before stop: status=ACTIVE  attempt=1  sdk_messages=5  response=NULL
after stop:  status=ACTIVE  attempt=1  sdk_messages=7  response=NULL
```

While compute was stopped, retrieval returned HTTP `503`. The runtime row and
SDK history remained in Lakebase. After `databricks apps start`, the scanner
claimed the stale row:

```text
after restart: status=ACTIVE     attempt=2  sdk_messages=12  response=NULL
final:         status=COMPLETED  attempt=2  sdk_messages=31  response=persisted
```

SDK message `112` was the fixed recovery note. It contained neither the PR URL
nor the old temporary workspace. The following recovered tool calls nevertheless
used `/tmp/openai-sdk-agent-0oxbysef/repo` and checked out PR 459. Those values
were present only in messages `105` through `111`, proving that attempt `2`
reopened and used the persisted SDK session history. The old pod-local directory
was gone, so the agent recreated it.

After completion:

- `GET /responses/runtime-crash-20260819T222154Z` returned HTTP `200`.
- Reposting the exact background request returned the cached response in 0.221 seconds.
- Retrieved and cached responses had the same SHA-256:
  `4929b07e0931a5a487efac00f4d13910b4ccf2d15e3edb470f670775eb34f5bc`.

## Result

The live tests verify the intended separation:

- `DatabricksDurableRuntime` persists request, response, status, attempt, and heartbeat.
- `AsyncDatabricksSession` persists replayable agent and tool history.
- Recovery replays the persisted request to the executor, while this executor
  intentionally resumes the agent with only the SDK session and recovery note.
- A stable execution ID lets a client poll after a disconnect or App restart.
- Pod-local files and in-flight tool processes are not durable.
