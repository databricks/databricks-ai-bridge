from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from databricks_ai_bridge.session_store import (
    DatabricksSessionStoreClient,
    SessionStoreError,
)


def _workspace_client() -> MagicMock:
    workspace = MagicMock()
    workspace.config.host = "https://workspace.example.com/"
    workspace.config.authenticate.return_value = {"Authorization": "Bearer token"}
    return workspace


@pytest.mark.asyncio
async def test_uses_workspace_auth_traffic_id_and_url_encoding() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"session_id": "a/b"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DatabricksSessionStoreClient(
        workspace_client=_workspace_client(),
        traffic_id="testenv://liteswap/session-store",
        http_client=http_client,
    )

    assert await client.get_session("a/b") == {"session_id": "a/b"}
    request = requests[0]
    assert str(request.url) == (
        "https://workspace.example.com/api/2.0/agent-conversation/sessions/a%2Fb"
    )
    assert request.headers["Authorization"] == "Bearer token"
    assert request.headers["x-databricks-traffic-id"] == ("testenv://liteswap/session-store")

    await client.aclose()
    assert not http_client.is_closed
    await http_client.aclose()


@pytest.mark.asyncio
async def test_ensure_session_returns_existing_session_after_conflict() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(409, json={"error": "exists"})
        return httpx.Response(200, json={"session_id": "existing"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DatabricksSessionStoreClient(
        workspace_client=_workspace_client(), http_client=http_client
    )

    session = await client.ensure_session(
        "existing",
        display_name="Support",
        metadata={"app": "demo"},
        user_id="user-1",
        parent_session="sessions/parent",
    )

    assert session == {"session_id": "existing"}
    assert [request.method for request in requests] == ["POST", "GET"]
    create_request = requests[0]
    assert create_request.url.params["session_id"] == "existing"
    assert create_request.read() == (
        b'{"display_name":"Support","metadata":{"app":"demo"},'
        b'"user_id":"user-1","parent_session":"sessions/parent"}'
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_list_methods_follow_page_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page_token = request.url.params.get("page_token")
        if request.url.path.endswith("/events"):
            if page_token is None:
                return httpx.Response(
                    200,
                    json={
                        "events": [{"sequence": 1, "data": {"value": 1}}],
                        "next_page_token": "event-page-2",
                    },
                )
            assert page_token == "event-page-2"
            return httpx.Response(200, json={"events": [{"sequence": 2, "data": {"value": 2}}]})
        if page_token is None:
            return httpx.Response(
                200,
                json={
                    "sessions": [{"session_id": "one"}],
                    "next_page_token": "session-page-2",
                },
            )
        assert page_token == "session-page-2"
        return httpx.Response(200, json={"sessions": [{"session_id": "two"}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DatabricksSessionStoreClient(
        workspace_client=_workspace_client(), http_client=http_client
    )

    sessions = await client.list_sessions(
        user_id="user-1",
        parent_session="sessions/parent",
        root_session="sessions/root",
        filter_expression='metadata.app = "demo"',
        order_by="update_time desc",
        page_size=25,
    )
    events = await client.list_events("one", page_size=20)

    assert [session["session_id"] for session in sessions] == ["one", "two"]
    assert [event["sequence"] for event in events] == [1, 2]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_event_mutations_and_idempotency_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("events:append"):
            return httpx.Response(200, json={"events": [{"event_id": "event-1"}]})
        if request.url.path.endswith("events:pop"):
            return httpx.Response(200, json={"event": {"event_id": "event-1"}})
        return httpx.Response(200, json={})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DatabricksSessionStoreClient(
        workspace_client=_workspace_client(), http_client=http_client
    )

    appended = await client.append_events(
        "session-1",
        [{"type": "message", "data": {"role": "user"}}],
        idempotency_key="append-1",
    )
    popped = await client.pop_event("session-1")
    await client.clear_events("session-1")
    await client.delete_session("session-1", force=True)

    assert appended == [{"event_id": "event-1"}]
    assert popped == {"event_id": "event-1"}
    assert requests[0].headers["Idempotency-Key"] == "append-1"
    assert requests[-1].url.params["force"] == "true"
    await http_client.aclose()


@pytest.mark.asyncio
async def test_raises_structured_error() -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"error": "bad filter"})
        )
    )
    client = DatabricksSessionStoreClient(
        workspace_client=_workspace_client(), http_client=http_client
    )

    with pytest.raises(SessionStoreError) as error:
        await client.list_sessions(filter_expression="bad")

    assert error.value.status_code == 400
    assert error.value.detail == {"error": "bad filter"}
    await http_client.aclose()
