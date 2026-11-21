"""Offline tests for the hosted dispatcher App in ``app/`` (no Databricks or Anthropic calls)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The App's modules live at the App root, which is how the Apps runtime imports them (`uvicorn app:app`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
import service as svc  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import SECURITY_HEADERS, create_app  # noqa: E402
from databricks_anthropic.sandbox.dispatcher import DispatcherStatus  # noqa: E402

ENV = {
    "ANTHROPIC_ENVIRONMENT_ID": "env_S",
    "ANTHROPIC_ENVIRONMENT_KEY": "sk-env-SECRET",
    "CMA_SANDBOX_ID_KEY": "hmac-SECRET",
}


def launcher(**env):
    return svc.build_launcher({**ENV, **env}, workspace_client=MagicMock())


class FakeDispatcher:
    """Polls until cancelled (like the real one), or crashes ``crash`` times first."""

    def __init__(self, launcher, *, max_concurrent_launches, crash=0):
        self.launcher, self.max_concurrent_launches, self.crash = (
            launcher,
            max_concurrent_launches,
            crash,
        )
        self.status = DispatcherStatus()
        self.runs = 0

    async def run(self):
        self.runs += 1
        if self.runs <= self.crash:
            raise ConnectionError("network down: sk-env-SECRET")  # message must never be surfaced
        self.status.polling = True
        try:
            await asyncio.Event().wait()
        finally:
            self.status.polling = False


def fake_service(crash=0, **env):
    return svc.DispatcherService(
        launcher(**env),
        make_dispatcher=lambda lc, **kw: FakeDispatcher(lc, crash=crash, **kw),
    )


# ---------------------------------------------------------------- configuration


def test_missing_configuration_names_keys_but_not_values():
    with pytest.raises(svc.ConfigError) as e:
        svc.build_launcher(
            {"ANTHROPIC_ENVIRONMENT_KEY": "sk-env-SECRET"}, workspace_client=MagicMock()
        )
    assert "ANTHROPIC_ENVIRONMENT_ID" in str(e.value) and "CMA_SANDBOX_ID_KEY" in str(e.value)
    assert "SECRET" not in str(e.value)


def test_launcher_uses_hmac_names_prefix_and_https():
    lc = launcher()
    assert lc.sandbox_id_key == "hmac-SECRET" and lc.sandbox_id_prefix == "cmaapp-"
    assert lc.sandbox_id_for("sesn_1") != svc.DatabricksSandboxLauncher(
        workspace_client=MagicMock(),
        environment_id="env_S",
        environment_key="k",
        sandbox_id_prefix="cmaapp-",
    ).sandbox_id_for("sesn_1")
    with pytest.raises(ValueError, match="https"):
        launcher(ANTHROPIC_BASE_URL_OVERRIDE="http://proxy")


def test_unknown_mcp_is_rejected():
    with pytest.raises(svc.ConfigError):
        launcher(CMA_MCP="something-else")


def test_genie_one_server_is_configured(monkeypatch):
    import databricks_anthropic.sandbox.mcp as mcp

    monkeypatch.setattr(mcp, "genie_one_mcp_url", lambda w: "https://h/api/2.0/mcp/genie")
    assert [s.name for s in launcher(CMA_MCP="genie-one").mcp_servers] == ["genie-one"]


# ---------------------------------------------------------------- routes


def test_routes_report_health_and_counters_without_secrets():
    with TestClient(create_app(lambda: fake_service())) as client:
        for _ in range(50):
            if client.get("/api/health").status_code == 200:
                break
        health = client.get("/api/health")
        assert health.status_code == 200 and health.json() == {"healthy": True}
        status = client.get("/api/status")
        body = status.text + client.get("/").text
        assert status.json()["environment_id"] == "env_S" and status.json()["mode"] == "service"
        assert "SECRET" not in body
        for name, value in SECURITY_HEADERS.items():
            assert status.headers[name] == value
        assert (
            client.get("/docs").status_code == 404
            and client.get("/openapi.json").status_code == 404
        )


def test_no_write_routes():
    with TestClient(create_app(lambda: fake_service())) as client:
        for path in ("/", "/api/status", "/api/health", "/api/sessions"):
            assert client.post(path, json={}).status_code in (404, 405)


@pytest.mark.asyncio
async def test_crash_loop_restarts_and_turns_unhealthy(monkeypatch):
    monkeypatch.setattr(svc, "RESTART_MIN_S", 0.01)
    monkeypatch.setattr(svc, "RESTART_MAX_S", 0.01)
    service = fake_service(crash=10)
    service.start()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if service.restarts >= svc.UNHEALTHY_AFTER_RESTARTS:
            break
    assert not service.healthy
    status = service.status()
    assert status["last_crash"] == "ConnectionError" and "SECRET" not in str(status)
    await service.stop()


@pytest.mark.asyncio
async def test_recovers_after_transient_crash(monkeypatch):
    monkeypatch.setattr(svc, "RESTART_MIN_S", 0.01)
    service = fake_service(crash=1)
    service.start()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if service.healthy:
            break
    assert service.healthy and service.restarts == 1
    await service.stop()
    assert not service.healthy


def test_status_shape_is_counters_only():
    service = fake_service()
    service.dispatcher = SimpleNamespace(status=DispatcherStatus(claimed=3, launched=2, failed=1))
    s = service.status()
    assert s["dispatcher"]["claimed"] == 3 and set(s) >= {"healthy", "restarts", "uptime_s"}


def test_sandbox_counts_by_state_and_failure_is_contained(monkeypatch):
    service = fake_service()
    sbs = [
        SimpleNamespace(status=SimpleNamespace(state=SimpleNamespace(value=v)))
        for v in ("SANDBOX_STATE_RUNNING", "SANDBOX_STATE_RUNNING", "SANDBOX_STATE_STOPPED")
    ]
    monkeypatch.setattr(svc.DatabricksSandboxDispatcher, "list_session_sandboxes", lambda self: sbs)
    assert service.sandbox_counts() == {"RUNNING": 2, "STOPPED": 1}

    def boom(self):
        raise PermissionError("denied: sk-env-SECRET")

    monkeypatch.setattr(svc.DatabricksSandboxDispatcher, "list_session_sandboxes", boom)
    assert service.sandbox_counts() == {"RUNNING": 2, "STOPPED": 1}  # cached
    assert service.sandbox_counts(max_age_s=0) == "unavailable (PermissionError)"
