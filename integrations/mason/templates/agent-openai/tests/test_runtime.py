from databricks_mason import DurableAgentApp
from databricks_mason.runtime.store import InMemoryDurabilityStore


async def _invoke(request, context):
    return request


def test_invocation_routes_support_local_and_deployed_app_auth_paths() -> None:
    server = DurableAgentApp(_invoke, durability_store=InMemoryDurabilityStore())
    paths = server.app.openapi()["paths"]

    assert paths["/invocations"]["post"]
    assert paths["/api/invocations"]["post"]
    assert paths["/invocations/{run_id}"]["get"]
    assert paths["/api/invocations/{run_id}"]["get"]
    assert paths["/invocations/{run_id}/events"]["get"]
    assert paths["/api/invocations/{run_id}/events"]["get"]
    assert "/api/session/new" not in paths
