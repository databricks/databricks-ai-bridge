"""Unit tests for `mason memory` store rendering (timestamp field mapping)."""

from __future__ import annotations

from click.testing import CliRunner

from databricks_mason import memory as memory_mod
from databricks_mason.memory import stores


class _Client:
    host = "https://example.databricks.com"

    def __init__(self, store=None, page=None):
        self._store = store
        self._page = page

    def get_memory_store(self, name):
        return self._store

    def create_memory_store(self, display_name, description=None):
        return self._store

    def list_memory_stores(self, page_size=None, page_token=None):
        return self._page


class _Ctx:
    def __init__(self, client, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


def test_store_created_updated_read_current_and_legacy_fields():
    # Current API field.
    assert (
        memory_mod._store_created({"create_time": "2026-08-15T01:29:00Z"}) == "2026-08-15T01:29:00Z"
    )
    assert (
        memory_mod._store_updated({"update_time": "2026-08-16T01:29:00Z"}) == "2026-08-16T01:29:00Z"
    )
    # Legacy epoch-millis fallback.
    assert memory_mod._store_created({"created_at": 1_755_100_000_000}) == 1_755_100_000_000
    assert memory_mod._store_updated({"updated_at": 1_755_100_000_000}) == 1_755_100_000_000
    # create_time takes precedence over the legacy field.
    assert memory_mod._store_created({"create_time": "new", "created_at": 1}) == "new"
    # Missing -> falsy (renders as em-dash downstream).
    assert not memory_mod._store_created({})


def test_store_get_renders_timestamps_from_create_time():
    store = {
        "name": "memory-stores/abc123",
        "display_name": "demo",
        "create_time": "2026-08-15T01:29:00Z",
        "update_time": "2026-08-16T01:29:00Z",
    }
    result = CliRunner().invoke(stores, ["get", "abc123"], obj=_Ctx(_Client(store=store)))
    assert result.exit_code == 0, result.output
    # Timestamps render (year present) instead of the em-dash placeholder.
    assert "2026" in result.output


def test_store_create_suggests_binding_the_store():
    store = {"name": "memory-stores/abc123", "display_name": "demo"}
    result = CliRunner().invoke(
        stores, ["create", "--display-name", "demo"], obj=_Ctx(_Client(store=store))
    )
    assert result.exit_code == 0, result.output
    assert "mason memory bind demo" in result.output


def test_store_list_renders_timestamps_from_create_time():
    page = {
        "managed_memory_stores": [
            {
                "name": "memory-stores/abc123",
                "display_name": "demo",
                "create_time": "2026-08-15T01:29:00Z",
                "update_time": "2026-08-15T01:29:00Z",
            }
        ]
    }
    result = CliRunner().invoke(stores, ["list"], obj=_Ctx(_Client(page=page)))
    assert result.exit_code == 0, result.output
    # A humanized relative time appears rather than "—" for the row.
    assert "ago" in result.output or "just now" in result.output


def _bind_ctx(tmp_path):
    """A CLI context whose client records memory-store creation, over a scaffolded agent.toml."""
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\n', encoding="utf-8"
    )

    class _BindClient:
        host = "https://example.databricks.com"

        def __init__(self):
            self.created = []

        def create_memory_store(self, display_name, description=None, *, retry_transient=False):
            self.created.append(display_name)
            # The API returns an id distinct from the display name, as the real one does.
            return {"name": "memory-stores/mem-id-123", "display_name": display_name}

        def list_memory_stores(self, page_size=None, page_token=None):
            return {"managed_memory_stores": []}

    return _Ctx(_BindClient())


def test_memory_bind_writes_agent_toml_and_creates_store(tmp_path):
    from databricks_mason.agent_project import AgentProject
    from databricks_mason.memory import memory as memory_group

    ctx = _bind_ctx(tmp_path)
    result = CliRunner().invoke(
        memory_group, ["bind", "agent-mem", "--source", str(tmp_path)], obj=ctx
    )

    assert result.exit_code == 0, result.output
    assert ctx.client().created == ["agent-mem"]  # created by default
    project = AgentProject.load(tmp_path)
    assert project.memory_store == "agent-mem"
    assert project.memory_store_id == "mem-id-123"  # bare id recorded for the runtime


def test_memory_unbind_clears_agent_toml(tmp_path):
    from databricks_mason.agent_project import AgentProject
    from databricks_mason.memory import memory as memory_group

    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\n\n[memory_store]\nname = "m"\n',
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        memory_group, ["unbind", "--source", str(tmp_path)], obj=_Ctx(_Client())
    )

    assert result.exit_code == 0, result.output
    assert AgentProject.load(tmp_path).memory_store is None
