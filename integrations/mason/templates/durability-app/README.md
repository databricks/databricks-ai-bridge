# Durability App

A minimal LangGraph application hosted by `databricks_mason.DurableAgentApp`.
It has no model dependency: the graph waits for an optional delay and returns a
deterministic result, which keeps the durability behavior easy to inspect.

## Run locally

```bash
uv sync
uv run start-server
```

Submit a background run and keep the routing cookie for polling:

```bash
COOKIE_JAR=/tmp/mason-durability.cookies

curl -sS -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -X POST http://localhost:8000/invocations \
  -H 'content-type: application/json' \
  -d '{
    "id": "run-1",
    "background": true,
    "input": {"message": "hello", "delay_seconds": 2}
  }'

curl -sS -b "$COOKIE_JAR" http://localhost:8000/invocations/run-1
curl -N -b "$COOKIE_JAR" http://localhost:8000/invocations/run-1/events
```

The client owns the invocation `id`. Retrying the same request with the same ID
returns the persisted run; reusing the ID with a different payload returns `409`.

## Deploy

Bare `mason init` scaffolds this template. Deploy it with an explicit profile:

```bash
mason --profile <profile> deploy durability-app --source .
```

Mason attaches one Lakebase database for the runtime tables. If `--session` is
provided, the runtime uses that Session Store database; otherwise Mason reuses or
provisions `<app>-durability`. Existing Mason templates are unaffected.

If an active run becomes stale after a process restart, the runtime claims a new
attempt and calls the configured recovery callback. This example uses the same
deterministic graph for initial and recovery attempts and returns `recovered: true`
when the attempt number is greater than one.
