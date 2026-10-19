"""Browser-origin configuration and the public invocation transport."""

import json
from uuid import uuid4

import httpx
import pytest

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentkit import DurableAgentServer
from databricks_agentkit.runtime.auth import InvocationAuthPolicy
from databricks_agentkit.runtime.cors import (
    CORS_ALLOWED_ORIGINS_ENV,
    allowed_origins_from_environment,
    validate_allowed_origins,
)
from databricks_agentkit.runtime.store import InMemoryRuntimeStore

ORIGIN = "https://chat.example.com"


@pytest.mark.parametrize(
    "origins",
    [
        "https://chat.example.com",
        [None],
        [1],
        ["*"],
        ["null"],
        ["ftp://chat.example.com"],
        ["https://chat.example.com/"],
        ["https://chat.example.com/path"],
        ["https://user:password@chat.example.com"],
        ["https://chat.example.com?"],
        ["https://chat.example.com#"],
        ["https://*.example.com"],
        ["https://chat.example.com:65536"],
        [" https://chat.example.com"],
        ["https://chat.example.com\n"],
        ["https://chat.example.com\\other"],
        ["https://chat.example.com\x00"],
    ],
)
def test_invalid_origins_fail_closed(origins):
    with pytest.raises(ValueError, match="origin"):
        validate_allowed_origins(origins)


def test_origins_are_normalized_and_deduplicated():
    assert validate_allowed_origins(
        [
            "https://CHAT.example.com:443",
            ORIGIN,
            "http://localhost:3000",
            "https://[::1]:8443",
        ]
    ) == (ORIGIN, "http://localhost:3000", "https://[::1]:8443")


@pytest.mark.parametrize("raw", ["not-json", '"*"', '["*"]', '["https://host/path"]'])
def test_invalid_runtime_environment_is_rejected(monkeypatch, raw):
    monkeypatch.setenv(CORS_ALLOWED_ORIGINS_ENV, raw)
    with pytest.raises(ValueError, match=CORS_ALLOWED_ORIGINS_ENV):
        allowed_origins_from_environment()


def test_manifest_round_trip_preserves_comments_and_can_clear_origins(tmp_path):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.write()
    project.path.write_text(project.path.read_text() + "\n# Keep this project comment.\n")
    project = AgentProject.load(tmp_path)
    project.set_allowed_origins((ORIGIN,))
    project.write()
    reloaded = AgentProject.load(tmp_path)
    assert reloaded.allowed_origins == (ORIGIN,)
    assert "# Keep this project comment." in project.path.read_text()
    reloaded.set_allowed_origins(())
    reloaded.write()
    assert AgentProject.load(tmp_path).allowed_origins == ()


@pytest.mark.parametrize("cors", ['cors = "invalid"', '[cors]\nallowed_origin = "https://host"'])
def test_malformed_manifest_cors_is_rejected(tmp_path, cors):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.write()
    text = project.path.read_text()
    project.path.write_text(text + "\n" + cors if cors.startswith("[") else cors + "\n" + text)
    with pytest.raises(AgentCliError, match="must be a table|Unsupported CORS settings"):
        AgentProject.load(tmp_path)


def test_custom_server_rejects_managed_cors(tmp_path):
    project = AgentProject.create(tmp_path, framework="langgraph", server="custom")
    with pytest.raises(AgentCliError, match="server = 'agentbricks'"):
        project.set_allowed_origins((ORIGIN,))


def make_server(**kwargs):
    server = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(),
        auth_policy=InvocationAuthPolicy(),
        **kwargs,
    )

    @server.invoke
    async def echo(value, context):
        return value

    @server.get("/_test/error")
    async def error():
        raise RuntimeError("test error")

    return server


@pytest.mark.asyncio
async def test_preflight_and_json_sse_status_and_errors(monkeypatch):
    monkeypatch.setenv(CORS_ALLOWED_ORIGINS_ENV, json.dumps([ORIGIN]))
    server = make_server()
    async with server.router.lifespan_context(server):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server, raise_app_exceptions=False),
            base_url="https://agent.example.com",
        ) as client:
            preflight = await client.options(
                "/api/invocations",
                headers={
                    "Origin": ORIGIN,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "authorization,content-type,x-routing-key",
                },
            )
            assert preflight.status_code == 200
            assert preflight.headers["access-control-allow-origin"] == ORIGIN
            assert "access-control-allow-credentials" not in preflight.headers
            assert "origin" in preflight.headers["vary"].lower()

            invocation_id = str(uuid4())
            headers = {"Origin": ORIGIN}
            response = await client.post(
                "/api/invocations",
                headers=headers,
                json={
                    "id": invocation_id,
                    "input": {"message": "hello"},
                    "stream": True,
                },
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert "run.completed" in response.text
            assert response.headers["access-control-allow-origin"] == ORIGIN

            status = await client.get(f"/api/invocations/{invocation_id}", headers=headers)
            assert status.json()["output"] == {"message": "hello"}
            assert status.headers["access-control-allow-origin"] == ORIGIN
            events = await client.get(f"/api/invocations/{invocation_id}/events", headers=headers)
            assert "run.completed" in events.text
            assert events.headers["access-control-allow-origin"] == ORIGIN
            sync = await client.post(
                "/api/invocations",
                headers=headers,
                json={
                    "id": str(uuid4()),
                    "input": {"message": "sync"},
                },
            )
            assert sync.status_code == 200
            assert sync.json()["output"] == {"message": "sync"}
            assert sync.headers["access-control-allow-origin"] == ORIGIN

            for response, expected in [
                (await client.post("/api/invocations", headers=headers, json={}), 422),
                (await client.get(f"/api/invocations/{uuid4()}", headers=headers), 404),
                (await client.get("/_test/error", headers=headers), 500),
            ]:
                assert response.status_code == expected
                assert response.headers["access-control-allow-origin"] == ORIGIN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin,method,requested_headers",
    [
        ("https://unlisted.example.com", "POST", "authorization"),
        (ORIGIN, "DELETE", "authorization"),
        (ORIGIN, "POST", "x-unapproved-header"),
    ],
)
async def test_disallowed_preflight_never_invokes_agent(origin, method, requested_headers):
    server = make_server(allowed_origins=[ORIGIN])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server),
        base_url="https://agent.example.com",
    ) as client:
        response = await client.options(
            "/api/invocations",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": requested_headers,
            },
        )
        assert response.status_code == 400
        if origin != ORIGIN:
            assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_cors_does_not_bypass_request_user_auth(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "test-agent")
    monkeypatch.delenv("DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL", raising=False)
    server = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(),
        auth_policy=InvocationAuthPolicy(user_tools=("test",)),
        allowed_origins=[ORIGIN],
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server),
        base_url="https://agent.example.com",
    ) as client:
        preflight = await client.options(
            "/api/invocations",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert preflight.status_code == 200
        response = await client.post(
            "/api/invocations",
            headers={"Origin": ORIGIN},
            json={
                "id": str(uuid4()),
                "input": {},
            },
        )
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == ORIGIN


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_no_origin_requests_keep_existing_behavior(monkeypatch, enabled):
    monkeypatch.delenv(CORS_ALLOWED_ORIGINS_ENV, raising=False)
    server = make_server(allowed_origins=[ORIGIN] if enabled else [])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server),
        base_url="https://agent.example.com",
    ) as client:
        response = await client.get("/not-found")
    assert response.status_code == 404
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_explicit_empty_origins_override_environment(monkeypatch):
    monkeypatch.setenv(CORS_ALLOWED_ORIGINS_ENV, json.dumps([ORIGIN]))
    server = make_server(allowed_origins=[])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server),
        base_url="https://agent.example.com",
    ) as client:
        response = await client.options(
            "/api/invocations",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code == 405
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_cors_is_disabled_when_not_configured(monkeypatch):
    monkeypatch.delenv(CORS_ALLOWED_ORIGINS_ENV, raising=False)
    server = make_server()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server),
        base_url="https://agent.example.com",
    ) as client:
        response = await client.options(
            "/api/invocations",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code == 405
    assert "access-control-allow-origin" not in response.headers
