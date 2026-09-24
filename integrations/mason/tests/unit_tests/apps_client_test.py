"""Direct unit tests for AppsClient - the databricks apps read/inspect helper."""

from __future__ import annotations

import json
import types

import pytest

import databricks_mason.apps_client as apps_client_mod
from databricks_mason.apps_client import AppsClient
from databricks_mason.errors import AgentCliError


def _ok(stdout=""):
    """Return a fake runner response with returncode 0 and the given stdout."""
    return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def _err():
    """Return a fake runner response with returncode 1."""
    return types.SimpleNamespace(returncode=1, stdout="", stderr="error")


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


def test_exists_true_when_rc_zero():
    calls = []

    def fake_run(args, profile, **kwargs):
        calls.append((args, profile, kwargs))
        return _ok()

    client = AppsClient("myprofile", runner=fake_run)
    assert client.exists("myapp") is True
    assert calls == [(["apps", "get", "myapp"], "myprofile", {"capture": True, "check": False})]


def test_exists_false_when_rc_nonzero():
    def fake_run(args, profile, **kwargs):
        return _err()

    client = AppsClient("myprofile", runner=fake_run)
    assert client.exists("myapp") is False


# ---------------------------------------------------------------------------
# service_principal
# ---------------------------------------------------------------------------


def test_service_principal_parses_client_id():
    calls = []

    def fake_run(args, profile, **kwargs):
        calls.append((args, profile))
        return _ok(json.dumps({"service_principal_client_id": "sp-abc"}))

    client = AppsClient("testprofile", runner=fake_run)
    result = client.service_principal("myapp")
    assert result == "sp-abc"
    assert calls == [(["apps", "get", "myapp", "-o", "json"], "testprofile")]


def test_service_principal_returns_none_on_rc_nonzero():
    def fake_run(args, profile, **kwargs):
        return _err()

    client = AppsClient("prof", runner=fake_run)
    assert client.service_principal("myapp") is None


def test_service_principal_returns_none_on_invalid_json():
    def fake_run(args, profile, **kwargs):
        return _ok("not-valid-json{")

    client = AppsClient("prof", runner=fake_run)
    assert client.service_principal("myapp") is None


# ---------------------------------------------------------------------------
# url
# ---------------------------------------------------------------------------


def test_url_parses_url_field():
    def fake_run(args, profile, **kwargs):
        return _ok(json.dumps({"url": "https://myapp.databricksapps.com"}))

    client = AppsClient("prof", runner=fake_run)
    assert client.url("myapp") == "https://myapp.databricksapps.com"


def test_url_returns_none_when_key_missing():
    def fake_run(args, profile, **kwargs):
        return _ok(json.dumps({"name": "agent-mason-myapp"}))

    client = AppsClient("prof", runner=fake_run)
    assert client.url("myapp") is None


def test_url_returns_none_when_url_empty_string():
    def fake_run(args, profile, **kwargs):
        return _ok(json.dumps({"url": ""}))

    client = AppsClient("prof", runner=fake_run)
    assert client.url("myapp") is None


def test_url_returns_none_on_rc_nonzero():
    def fake_run(args, profile, **kwargs):
        return _err()

    client = AppsClient("prof", runner=fake_run)
    assert client.url("myapp") is None


# ---------------------------------------------------------------------------
# compute_state
# ---------------------------------------------------------------------------


def test_compute_state_parses_nested_state():
    calls = []

    def fake_run(args, profile, **kwargs):
        calls.append((args, profile))
        return _ok(json.dumps({"compute_status": {"state": "ACTIVE"}}))

    client = AppsClient("prof", runner=fake_run)
    assert client.compute_state("myapp") == "ACTIVE"
    assert calls == [(["apps", "get", "myapp", "-o", "json"], "prof")]


def test_compute_state_returns_none_when_compute_status_missing():
    def fake_run(args, profile, **kwargs):
        return _ok(json.dumps({"name": "agent-mason-myapp"}))

    client = AppsClient("prof", runner=fake_run)
    assert client.compute_state("myapp") is None


def test_compute_state_returns_none_on_rc_nonzero():
    def fake_run(args, profile, **kwargs):
        return _err()

    client = AppsClient("prof", runner=fake_run)
    assert client.compute_state("myapp") is None


# ---------------------------------------------------------------------------
# wait_for_running (moved from deploy_test.py)
# ---------------------------------------------------------------------------


def test_wait_for_running_returns_when_compute_active():
    def fake_run(args, profile, **kwargs):
        return _ok(json.dumps({"compute_status": {"state": "ACTIVE"}}))

    client = AppsClient("prof", runner=fake_run)
    client.wait_for_running("app", timeout_s=1)  # returns without raising


def test_wait_for_running_times_out(monkeypatch):
    def fake_run(args, profile, **kwargs):
        return _ok(json.dumps({"compute_status": {"state": "STARTING"}}))

    monkeypatch.setattr(apps_client_mod.time, "sleep", lambda s: None)  # don't actually wait
    client = AppsClient("prof", runner=fake_run)
    with pytest.raises(AgentCliError):
        client.wait_for_running("app", timeout_s=0)
