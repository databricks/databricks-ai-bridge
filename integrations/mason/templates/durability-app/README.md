# Durability App

A minimal LangGraph application hosted by `databricks_mason.DurableAgentApp`.
It has no model dependency: the graph returns a deterministic result, which
keeps the durability behavior easy to inspect.

## Run locally

```bash
uv sync
uv run start-server
```

Submit a background run with one client-generated UUID and one stable routing
cookie. Databricks Apps supplies the cookie in deployment; for plain-HTTP
localhost `curl`, send it explicitly because the SDK marks it `Secure`:

```bash
ROUTING_COOKIE='__Host-databricks-app-router=11111111-1111-4111-8111-111111111111'
RUN_ID='22222222-2222-4222-8222-222222222222'

curl -sS -H "Cookie: $ROUTING_COOKIE" \
  -X POST http://localhost:8000/api/invocations \
  -H 'content-type: application/json' \
  -d "$(jq -nc --arg id "$RUN_ID" \
    '{id:$id,background:true,input:{message:"hello"}}')"

curl -sS -H "Cookie: $ROUTING_COOKIE" \
  "http://localhost:8000/api/invocations/$RUN_ID"
curl -N -H "Cookie: $ROUTING_COOKIE" \
  "http://localhost:8000/api/invocations/$RUN_ID/events"
```

The client owns the invocation `id`. Retrying the same request with the same ID
returns the persisted run; reusing the ID with a different payload returns `409`.

## Deploy

Bare `mason init` scaffolds this template. Deploy it with an explicit profile:

```bash
mason --profile <profile> deploy durability-app --source .
```

Bare `mason init` records the durability binding in `agent.toml`. At deploy time Mason
attaches one Lakebase database for the runtime tables, reusing the Session Store database
or otherwise reusing or provisioning `<app>-durability`. Runtime tables live in the app-owned
`databricks_mason_runtime_<app-hash>` schema. Existing Mason templates are unaffected.

If an active run becomes stale after a process restart, the runtime claims a new
attempt and calls the function registered with `@app.on_recovery`. This example
uses the same deterministic graph for initial and recovery attempts.
