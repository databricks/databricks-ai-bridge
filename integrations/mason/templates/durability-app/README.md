# Durability App

A minimal LangGraph application hosted by `databricks_mason.DurableAgentApp`.
It has no model dependency: the graph returns a deterministic result, which
keeps the durability behavior easy to inspect.

## Run locally

```bash
uv sync
uv run start-server
```

Submit a background run with one stable routing cookie for polling. Databricks
Apps supplies this cookie in deployment; for plain-HTTP localhost `curl`, set it
explicitly because the SDK correctly marks it `Secure`:

```bash
ROUTING_COOKIE='__Host-databricks-app-router=local-durability-session'

curl -sS -H "Cookie: $ROUTING_COOKIE" \
  -X POST http://localhost:8000/invocations \
  -H 'content-type: application/json' \
  -d '{
    "id": "run-1",
    "background": true,
    "input": {"message": "hello"}
  }'

curl -sS -H "Cookie: $ROUTING_COOKIE" \
  http://localhost:8000/invocations/run-1
curl -N -H "Cookie: $ROUTING_COOKIE" \
  http://localhost:8000/invocations/run-1/events
```

The client owns the invocation `id`. Retrying the same request with the same ID
returns the persisted run; reusing the ID with a different payload returns `409`.

## Deploy

Bare `mason init` scaffolds this template. Deploy it with an explicit profile:

```bash
mason --profile <profile> deploy durability-app --source .
```

Bare `mason init` records the durability binding in `agent.toml`; an existing Mason
project can opt in with `mason durability bind`. At deploy time Mason attaches one
Lakebase database for the runtime tables, reusing the Session Store database first,
then the Memory Store database, and otherwise reusing or provisioning
`<app>-durability`. Existing Mason templates are unaffected.

If an active run becomes stale after a process restart, the runtime claims a new
attempt and calls the configured recovery callback. This example uses the same
deterministic graph for initial and recovery attempts and returns `recovered: true`
when the attempt number is greater than one.
