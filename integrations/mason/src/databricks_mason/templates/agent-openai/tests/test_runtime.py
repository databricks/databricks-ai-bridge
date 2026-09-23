from databricks_mason import DurableAgentServer
from databricks_mason.runtime.store import InMemoryRuntimeStore


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
