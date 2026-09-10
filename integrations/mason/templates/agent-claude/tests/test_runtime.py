from databricks_mason import AgentApp
from databricks_mason.runtime.durability.store import InMemoryDurabilityStore


async def _invoke(request, context):
    return request


def test_runtime_exposes_only_durable_invocation_routes() -> None:
    app = AgentApp(durable_runtime=True, durability_store=InMemoryDurabilityStore())
    app.invoke(_invoke)
    app.on_recovery(_invoke)
    paths = app.openapi()["paths"]

    assert paths["/api/invocations"]["post"]
    assert paths["/api/invocations/{invocation_id}"]["get"]
    assert paths["/api/invocations/{invocation_id}/events"]["get"]
    assert "/invocations" not in paths
    assert "/api/health" not in paths
