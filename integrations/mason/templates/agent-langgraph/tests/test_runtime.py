from collections.abc import AsyncGenerator

from runtime.runtime import build_app


async def _invoke(request: dict) -> dict:
    return request


async def _stream(request: dict) -> AsyncGenerator[dict, None]:
    yield request


def test_invocation_routes_support_local_and_deployed_app_auth_paths() -> None:
    paths = build_app(_invoke, _stream).openapi()["paths"]

    assert paths["/invocations"]["post"]
    assert paths["/api/invocations"]["post"]
    assert paths["/invocations/{invocation_id}"]["get"]
    assert paths["/api/invocations/{invocation_id}"]["get"]
    assert paths["/health"]["get"]
    assert paths["/api/health"]["get"]
