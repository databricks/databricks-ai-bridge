from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("langgraph.checkpoint.postgres")

from databricks_ai_bridge import lakebase

from databricks_langchain import DatabricksStore


class TestConnectionPool:
    def __init__(self, connection_value="conn"):
        self.connection_value = connection_value
        self.conninfo = ""

    def __call__(
        self,
        *,
        conninfo,
        connection_class=None,
        **kwargs,
    ):
        self.conninfo = conninfo
        return self

    def connection(self):
        class _Ctx:
            def __init__(self, outer):
                self.outer = outer

            def __enter__(self):
                return self.outer.connection_value

            def __exit__(self, exc_type, exc, tb):
                pass

        return _Ctx(self)


def test_databricks_store_configures_lakebase(monkeypatch):
    mock_conn = MagicMock()
    test_pool = TestConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    from langgraph.store.postgres import PostgresStore

    monkeypatch.setattr(PostgresStore, "setup", MagicMock())

    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    store = DatabricksStore(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    )

    assert (
        test_pool.conninfo
        == "dbname=databricks_postgres user=test@databricks.com host=db-host port=5432 sslmode=require"
    )
    assert isinstance(store, DatabricksStore)

    with store._lakebase.connection() as conn:
        assert conn == mock_conn
