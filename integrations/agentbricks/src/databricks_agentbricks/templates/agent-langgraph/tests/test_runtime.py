from databricks_agentkit import DurableAgentServer
from databricks_agentkit.runtime.store import InMemoryRuntimeStore


async def _invoke(request, context):
    return request


def test_runtime_exposes_invocation_routes() -> None:
    app = DurableAgentServer(runtime_store=InMemoryRuntimeStore())
    app.invoke(_invoke)
    app.recover(_invoke)
    paths = app.openapi()["paths"]

    assert paths["/api/invocations"]["post"]
    assert paths["/api/invocations/{invocation_id}"]["get"]
    assert paths["/api/invocations/{invocation_id}/events"]["get"]
    assert "/invocations" not in paths
    assert "/api/health" not in paths


def test_runtime_requires_session_header() -> None:
    app = DurableAgentServer(runtime_store=InMemoryRuntimeStore())
    app.invoke(_invoke)
    request = app.openapi()["paths"]["/api/invocations"]["post"]
    session_header = next(
        parameter
        for parameter in request["parameters"]
        if parameter["name"].lower() == "x-databricks-session-id"
    )
    assert session_header["in"] == "header"
    assert session_header["required"] is True
