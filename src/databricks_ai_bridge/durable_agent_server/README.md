# Databricks Durable Agent Server

`DatabricksDurableAgentServer` extends MLflow `AgentServer` with the
request/response durability provided by `DatabricksDurableRuntime`.

Application authors keep the normal MLflow `@invoke()` handler. The server
reuses AgentServer request validation, response validation, tracing, request
header propagation, and optional chat proxy, and adds:

- blocking and background `POST /responses` and `POST /invocations`;
- `GET /responses/{execution_id}` status/result retrieval;
- Lakebase request/response persistence, heartbeats, and stale recovery; and
- `get_durable_execution_context()` for the current execution ID and attempt.

```python
from mlflow.genai.agent_server import invoke

from databricks_ai_bridge.durable_agent_server import (
    DatabricksDurableAgentServer,
    PreparedDurableRequest,
    get_durable_execution_context,
)


@invoke()
async def invoke_agent(request):
    context = get_durable_execution_context()
    if context.is_recovery:
        return await resume_agent(request)
    return await run_agent(request)


def prepare_request(request) -> PreparedDurableRequest:
    execution_id = request.custom_inputs["execution_id"]
    return PreparedDurableRequest(
        execution_id,
        request.model_dump(mode="json", exclude_none=True),
    )


server = DatabricksDurableAgentServer(
    prepare_request=prepare_request,
    schema="my_agent_durability",
)
app = server.app
```

The request preparer runs after AgentServer validation and returns the stable
execution ID plus normalized JSON persisted for retries. `background` and
`stream` are transport fields and are removed first. Streaming is currently
rejected because durable stream event storage is outside this runtime's
request/response contract.

## Inheritance tradeoff

The application gets AgentServer's standard `@invoke()` developer experience
and no longer renders in-progress Responses objects. Internally, the subclass
must replace AgentServer's private route setup because background mode must be
read before the ResponsesAgent validator discards transport-only fields. It
also calls the protected `_handle_invoke_request` hook to preserve AgentServer
tracing and output validation. This is less isolated from MLflow internals than
the standalone composition design.

Install with `databricks-ai-bridge[agent-server]`.
