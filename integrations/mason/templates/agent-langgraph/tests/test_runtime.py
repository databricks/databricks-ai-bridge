from databricks_mason.runtime import DurableAgentApp


def _app() -> DurableAgentApp:
    app = DurableAgentApp()

    @app.invoke
    async def invoke(request, context):
        return request

    @app.recover
    async def recover(request, context):
        return request

    return app


def test_invocation_routes_support_local_and_deployed_app_auth_paths() -> None:
    paths = _app().asgi_app.openapi()["paths"]

    assert paths["/invocations"]["post"]
    assert paths["/api/invocations"]["post"]
    assert paths["/invocations/{run_id}"]["get"]
    assert paths["/api/invocations/{run_id}"]["get"]
    assert paths["/invocations/{run_id}/events"]["get"]
    assert paths["/api/invocations/{run_id}/events"]["get"]
    assert paths["/health"]["get"]
    assert paths["/api/health"]["get"]
