"""Unit tests for StoreProvisioner: resolve, ensure, reconcile, and grant store operations."""

from __future__ import annotations

from unittest import mock

import pytest

from databricks_mason.errors import AgentCliError
from databricks_mason.store_provisioner import StoreProvisioner

# ---------------------------------------------------------------------------
# Minimal fake client for store-provisioner tests
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal managed-store client stub seeded with one pre-existing memory store.

    "mem" exists with id "mem-id-123" (id differs from display name, matching the real API).
    Created stores are appended so reconcile's auto-create can then resolve them.
    """

    def __init__(self):
        self._memory_stores = [{"name": "memory-stores/mem-id-123", "display_name": "mem"}]

    def list_memory_stores(self, page_size=None, page_token=None):
        return {"managed_memory_stores": list(self._memory_stores), "next_page_token": ""}

    def create_memory_store(self, display_name, *, retry_transient=False):
        for existing in self._memory_stores:
            if existing.get("display_name") == display_name:
                raise AgentCliError(
                    f"Memory store '{display_name}' already exists", error_code="ALREADY_EXISTS"
                )
        store = {"name": f"memory-stores/{display_name}", "display_name": display_name}
        self._memory_stores.append(store)
        return store

    def get_session_store(self, name):
        return {"session_store_name": name}

    def create_session_store(self, name, *, retry_transient=False):
        return {"session_store_name": name}

    def grant_session_store_permission(self, store, principal):
        return None

    def grant_memory_store_permission(self, store, principal):
        return None


# ---------------------------------------------------------------------------
# ensure_session_store
# ---------------------------------------------------------------------------


def test_ensure_session_store_reuses_on_already_exists():
    client = mock.Mock()
    client.create_session_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.get_session_store.return_value = {"session_store_name": "s"}
    # Reused store -> created is False.
    assert StoreProvisioner(client).ensure_session_store("s") == (
        {"session_store_name": "s"},
        False,
    )
    client.create_session_store.assert_called_once_with("s", retry_transient=True)


def test_ensure_session_store_reports_created():
    client = mock.Mock()
    client.create_session_store.return_value = {"session_store_name": "s"}
    assert StoreProvisioner(client).ensure_session_store("s") == ({"session_store_name": "s"}, True)


def test_ensure_session_store_permission_denied_gives_admin_hint():
    client = mock.Mock()
    client.create_session_store.side_effect = AgentCliError(
        "denied", error_code="PERMISSION_DENIED"
    )
    with pytest.raises(AgentCliError) as excinfo:
        StoreProvisioner(client).ensure_session_store("s")
    err = excinfo.value
    assert "permission to create session store 's'" in err.message
    assert err.hint is not None and "workspace admin" in err.hint


def test_ensure_session_store_already_exists_but_inaccessible():
    client = mock.Mock()
    client.create_session_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.get_session_store.side_effect = AgentCliError("denied", error_code="PERMISSION_DENIED")
    with pytest.raises(AgentCliError) as excinfo:
        StoreProvisioner(client).ensure_session_store("s")
    err = excinfo.value
    assert "already exists but you don't have access" in err.message
    assert err.hint is not None and "grant you access" in err.hint


# ---------------------------------------------------------------------------
# ensure_memory_store
# ---------------------------------------------------------------------------


def test_ensure_memory_store_reuses_on_already_exists():
    client = mock.Mock()
    client.create_memory_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.list_memory_stores.return_value = {
        "managed_memory_stores": [{"name": "memory-stores/mem-id-123", "display_name": "mem"}]
    }

    # Reused store -> created is False.
    assert StoreProvisioner(client).ensure_memory_store("mem") == (
        {"name": "memory-stores/mem-id-123", "display_name": "mem"},
        False,
    )
    client.create_memory_store.assert_called_once_with("mem", retry_transient=True)


def test_ensure_memory_store_reports_created():
    client = mock.Mock()
    client.create_memory_store.return_value = {"name": "memory-stores/mem-id-123"}
    assert StoreProvisioner(client).ensure_memory_store("mem") == (
        {"name": "memory-stores/mem-id-123"},
        True,
    )


def test_ensure_memory_store_permission_denied_gives_admin_hint():
    # ML-69282: admin-restricted Lakebase project creation -> actionable message, not a raw error.
    client = mock.Mock()
    client.create_memory_store.side_effect = AgentCliError("denied", error_code="PERMISSION_DENIED")
    with pytest.raises(AgentCliError) as excinfo:
        StoreProvisioner(client).ensure_memory_store("mem")
    err = excinfo.value
    assert "permission to create memory store 'mem'" in err.message
    assert err.hint is not None and "workspace admin" in err.hint
    assert "--no-create-stores" in err.hint


def test_ensure_memory_store_already_exists_but_inaccessible():
    # ML-69292: name taken but not visible to the caller -> "you don't have access", not "could
    # not be resolved".
    client = mock.Mock()
    client.create_memory_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.list_memory_stores.return_value = {"managed_memory_stores": []}
    with pytest.raises(AgentCliError) as excinfo:
        StoreProvisioner(client).ensure_memory_store("mem")
    err = excinfo.value
    assert "already exists but you don't have access" in err.message
    assert err.hint is not None and "grant you access" in err.hint


# ---------------------------------------------------------------------------
# resolve_memory_store
# ---------------------------------------------------------------------------


def test_resolve_memory_store_pages_at_100_and_matches_display_name():
    # The list API caps page_size at 100, so resolution must page (not request 1000) and match the
    # display name across pages.
    class _PagingClient:
        def __init__(self):
            self.calls = []

        def list_memory_stores(self, page_size=None, page_token=None):
            self.calls.append((page_size, page_token))
            if page_token is None:
                return {
                    "managed_memory_stores": [{"name": "memory-stores/a", "display_name": "other"}],
                    "next_page_token": "p2",
                }
            return {
                "managed_memory_stores": [{"name": "memory-stores/b", "display_name": "wanted"}],
                "next_page_token": "",
            }

    client = _PagingClient()
    store = StoreProvisioner(client).resolve_memory_store("wanted")
    assert store is not None
    assert store["name"] == "memory-stores/b"  # found on page 2
    assert all(ps == 100 for ps, _ in client.calls)  # never exceeds the API cap
    assert [pt for _, pt in client.calls] == [None, "p2"]  # followed the page token


def test_resolve_memory_store_returns_none_when_absent():
    class _EmptyClient:
        def list_memory_stores(self, page_size=None, page_token=None):
            return {"managed_memory_stores": [], "next_page_token": ""}

    assert StoreProvisioner(_EmptyClient()).resolve_memory_store("nope") is None


# ---------------------------------------------------------------------------
# grant_store_access
# ---------------------------------------------------------------------------


def test_grant_store_access_grants_both_stores_via_api(monkeypatch):
    # Grants go through the managed store API (the store service does the Lakebase grant server-side),
    # not a direct Lakebase resource attach - so a non-owner/non-admin deployer can still grant.
    calls = []

    class _Client:
        def grant_session_store_permission(self, name, sp):
            calls.append(("session", name, sp))

        def grant_memory_store_permission(self, name, sp):
            calls.append(("memory", name, sp))

    # Memory is granted by resource id, so the display-name binding is resolved first.
    monkeypatch.setattr(
        StoreProvisioner,
        "resolve_memory_store",
        lambda self, name: {"name": "memory-stores/uuid-x"},
    )
    err = StoreProvisioner(_Client()).grant_store_access("sp-1", "sess-1", "mem-display")

    assert err is None
    assert calls == [
        ("session", "sess-1", "sp-1"),
        ("memory", "memory-stores/uuid-x", "sp-1"),
    ]


def test_grant_store_access_surfaces_api_error(monkeypatch):
    class _Client:
        def grant_session_store_permission(self, name, sp):
            raise AgentCliError("grant failed", hint="the store service refused the grant")

    err = StoreProvisioner(_Client()).grant_store_access("sp", "sess-1", None)
    assert err == "the store service refused the grant"


def test_grant_store_access_memory_store_unresolvable(monkeypatch):
    # When the memory store display name can't be resolved (list returns nothing matching),
    # grant_store_access returns an error string instead of raising.
    class _Client:
        def grant_session_store_permission(self, name, sp):
            pass

    monkeypatch.setattr(StoreProvisioner, "resolve_memory_store", lambda self, name: None)
    err = StoreProvisioner(_Client()).grant_store_access("sp", "sess-1", "missing-store")
    assert err is not None
    assert "missing-store" in err
    assert "could not be resolved" in err


# ---------------------------------------------------------------------------
# reconcile_declared_stores
# ---------------------------------------------------------------------------


def test_reconcile_declared_stores_returns_none_when_unbound():
    assert StoreProvisioner(_FakeClient()).reconcile_declared_stores(None, None) is None


def test_reconcile_declared_stores_creates_missing_and_returns_memory_id(capsys):
    client = _FakeClient()  # seeded with only "mem" (id mem-id-123)
    memory_id = StoreProvisioner(client).reconcile_declared_stores("new-mem", "new-sess")
    # A freshly created memory store's bare id is returned for AGENT_MEMORY_STORE.
    assert memory_id == "new-mem"  # _FakeClient names created stores memory-stores/<display_name>
    out = capsys.readouterr().out
    assert "Created memory store 'new-mem'" in out
    assert "Created session store 'new-sess'" in out


def test_reconcile_declared_stores_reuses_existing_memory_id(capsys):
    client = _FakeClient()  # "mem" already exists with id mem-id-123
    memory_id = StoreProvisioner(client).reconcile_declared_stores("mem", None)
    assert memory_id == "mem-id-123"
    assert "Created memory store" not in capsys.readouterr().out  # reused, not created
