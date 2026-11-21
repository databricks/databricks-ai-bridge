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


def test_importing_or_apply_never_changes_os_environ(monkeypatch):
    import importlib
    import os

    for key in _useragent.UPSTREAM_ENV:
        monkeypatch.delenv(key, raising=False)
    before = dict(os.environ)
    _useragent.apply()
    importlib.reload(importlib.import_module("databricks_anthropic"))
    assert dict(os.environ) == before
    assert useragent.product() == (_useragent.PRODUCT_NAME, _useragent.PRODUCT_VERSION)


def test_subprocess_env_carries_the_upstream_variables_without_touching_os_environ(monkeypatch):
    import os

    for key in _useragent.UPSTREAM_ENV:
        monkeypatch.delenv(key, raising=False)
    env = _useragent.subprocess_env()
    assert {k: env[k] for k in _useragent.UPSTREAM_ENV} == _useragent.UPSTREAM_ENV
    assert not any(k in os.environ for k in _useragent.UPSTREAM_ENV)


def _load_worker():
    import importlib.util

    spec = importlib.util.find_spec("databricks_anthropic.sandbox.worker")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_memory_calls_carry_the_connector(sent, monkeypatch):
    """The in-sandbox memory tools call /api/2.0/agents/* through the worker's client: the product is the connector."""
    import asyncio

    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "t")
    monkeypatch.setenv(
        "CMA_PRODUCT_VERSION", _useragent.PRODUCT_VERSION
    )  # what the launcher passes
    worker = _load_worker()
    store = worker._MemoryStore(
        worker._workspace_client(), "cma-memory-abc", "databricks-user:1", "1"
    )
    with pytest.raises(Exception):  # noqa: B017 - Captured, possibly wrapped by the SDK
        asyncio.run(store.search("q", 1))
    assert sent and sent[-1]["User-Agent"].split()[0] == TOKEN


def test_memory_calls_on_a_launcher_tagged_client_carry_the_connector(sent):
    """The App's memory start-up check and the CLI's memory commands use clients the launcher (or login) tagged."""
    from databricks_anthropic.sandbox.launcher import DatabricksSandboxLauncher
    from databricks_anthropic.sandbox.memory import MEMORY_API

    w = client()  # WorkspaceClient(host=...): product unknown until tagged
    DatabricksSandboxLauncher(workspace_client=w, environment_id="env_1", environment_key="sk")
    with pytest.raises(Exception):  # noqa: B017
        w.api_client.do("GET", MEMORY_API, query={"page_size": 1})
    assert TOKEN in sent[-1]["User-Agent"].split()


def test_the_cli_client_sends_the_connector(sent, tmp_path, monkeypatch):
    """The CLI (run --memory, memory show/delete) builds clients with auth.workspace_client: product = connector."""
    from databricks_anthropic.cli import auth
    from databricks_anthropic.sandbox.memory import MEMORY_API

    cfg = tmp_path / "test.cfg"
    cfg.write_text("[uatest]\nhost = https://example.cloud.databricks.com\ntoken = t\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
    w = auth.workspace_client("uatest")
    with pytest.raises(Exception):  # noqa: B017
        w.api_client.do("GET", MEMORY_API, query={"page_size": 1})
    assert sent[-1]["User-Agent"].split()[0] == TOKEN
