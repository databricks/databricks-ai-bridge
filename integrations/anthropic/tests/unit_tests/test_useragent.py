"""Partner attribution reaches real HTTP requests (not just the client's config)."""

from __future__ import annotations

from typing import Any

import pytest
import requests
from databricks.sdk import WorkspaceClient, useragent
from databricks.sdk.core import Config

from databricks_anthropic import _useragent

TOKEN = f"{_useragent.PRODUCT_NAME}/{_useragent.PRODUCT_VERSION}"


class Captured(Exception):
    pass


@pytest.fixture
def sent(monkeypatch, tmp_path) -> list[dict[str, Any]]:
    """Capture outgoing request headers and stop before the network (no ambient profile or login)."""
    for key in [
        k for k in __import__("os").environ if k.startswith("DATABRICKS_") and "UPSTREAM" not in k
    ]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(tmp_path / "none.cfg"))
    seen: list[dict[str, Any]] = []

    def send(self, request, **kwargs):
        seen.append(dict(request.headers))
        raise Captured(request.url)

    monkeypatch.setattr(requests.Session, "send", send)
    return seen


def client() -> WorkspaceClient:
    return WorkspaceClient(host="https://example.cloud.databricks.com", token="t", auth_type="pat")


def call(w: WorkspaceClient) -> None:
    with pytest.raises(Exception):  # noqa: B017 - Captured, possibly wrapped by the SDK's retry/error layer
        w.sandbox.get_sandbox("sandboxes/x")


def test_importing_the_package_sets_the_global_product(sent):
    assert useragent.product() == (_useragent.PRODUCT_NAME, _useragent.PRODUCT_VERSION)
    # The partner-guide pattern: Config first, then WorkspaceClient(config=...).
    cfg = Config(host="https://example.cloud.databricks.com", token="t", auth_type="pat")
    call(WorkspaceClient(config=cfg))
    assert sent[0]["User-Agent"].split()[0] == TOKEN  # the product slot, not an extra


def test_plain_constructor_needs_tag(sent):
    w = client()  # WorkspaceClient(host=...) defaults its product to unknown/0.0.0
    call(w)
    assert TOKEN not in sent[-1]["User-Agent"].split()
    call(_useragent.tag(w))
    assert TOKEN in sent[-1]["User-Agent"].split()


def test_tag_reaches_requests_of_a_client_built_before_the_product_was_set(sent, monkeypatch):
    monkeypatch.setattr(useragent, "_product_name", "unknown")
    monkeypatch.setattr(useragent, "_product_version", "0.0.0")
    w = client()  # caches the User-Agent without our product
    call(w)
    assert TOKEN not in sent[-1]["User-Agent"].split()
    _useragent.tag(w)
    _useragent.tag(w)  # idempotent
    call(w)
    assert sent[-1]["User-Agent"].split().count(TOKEN) == 1


def test_apply_sets_upstream_env_for_cli_subprocesses(monkeypatch):
    for key in _useragent.UPSTREAM_ENV:
        monkeypatch.delenv(key, raising=False)
    _useragent.apply()
    import os

    assert {k: os.environ[k] for k in _useragent.UPSTREAM_ENV} == _useragent.UPSTREAM_ENV
