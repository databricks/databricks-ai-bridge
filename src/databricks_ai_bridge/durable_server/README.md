# Databricks Durable Server

`DatabricksDurableServer` adds a FastAPI transport to
`DatabricksDurableRuntime`. It owns application lifecycle and exposes:

- `POST /responses` and `POST /invocations` for blocking or background work;
- `GET /responses/{execution_id}` for status and result retrieval; and
- `GET /health` and `GET /api/healthz` for health checks.

The caller supplies three application-specific pieces:

1. `prepare_request`, which validates transport JSON and returns a stable
   execution ID plus the normalized JSON payload to persist;
2. an async executor, which owns agent sessions and recovery behavior; and
3. optionally, `status_response`, which maps non-completed state into the
   protocol-specific response shape.

```python
from databricks_ai_bridge.durable_server import (
    DatabricksDurableServer,
    PreparedDurableRequest,
)


def prepare_request(request: dict) -> PreparedDurableRequest:
    execution_id = request["custom_inputs"]["execution_id"]
    return PreparedDurableRequest(execution_id, request)


server = DatabricksDurableServer(
    execute,
    prepare_request=prepare_request,
    schema="my_agent_durability",
)
app = server.app
```

`background` and `stream` are transport fields and are removed before
`prepare_request` runs. Background requests return HTTP `202` until terminal;
completed cache hits return the stored response with HTTP `200`. Streaming is
currently rejected because durable stream event storage is outside this
runtime's request/response contract.

Install with `databricks-ai-bridge[agent-server]`.
