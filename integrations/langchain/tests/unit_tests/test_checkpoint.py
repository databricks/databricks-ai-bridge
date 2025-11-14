from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks_ai_bridge import lakebase

from databricks_langchain import CheckpointSaver

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("langgraph.checkpoint.postgres")


class RecordingConnectionPool:
    def __init__(self, log, connection_value="conn"):
        self.log = log
        self.connection_value = connection_value
        self.kwargs = None
        self.conninfo = ""
        self.connection_class = None
        self.getconn_calls = 0
        self.putconn_calls = []
        self.closed = False

    def __call__(
        self,
        *,
        conninfo,
        connection_class=None,
        kwargs,
        **extra,
    ):
        self.conninfo = conninfo
        self.connection_class = connection_class
        self.kwargs = kwargs
        return self

    def connection(self):
        self.log.append("pool_connection")

        class _Ctx:
            def __init__(self, outer):
                self.outer = outer
                self.entered = False
                self.exited = False

            def __enter__(self):
                self.outer.log.append("ctx_enter")
                self.entered = True
                return self.outer.connection_value

            def __exit__(self, exc_type, exc, tb):
                self.outer.log.append("ctx_exit")
                self.exited = True

        return _Ctx(self)

    def getconn(self):
        self.getconn_calls += 1
        return self.connection_value

    def putconn(self, conn):
        self.putconn_calls.append(conn)

    def close(self):
        self.closed = True


def test_checkpoint_saver_configures_lakebase(monkeypatch):
    log = []
    fake_pool = RecordingConnectionPool(log, connection_value="lake-conn")
    monkeypatch.setattr(lakebase, "ConnectionPool", fake_pool)

    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    saver = CheckpointSaver(
        database_instance="lakebase-instance",
        workspace_client=workspace,
    )

    assert (
        fake_pool.conninfo
        == "dbname=databricks_postgres user=test@databricks.com host=db-host port=5432 sslmode=require"
    )
    assert isinstance(saver, CheckpointSaver)

    with saver:
        pass

    assert fake_pool.putconn_calls == ["lake-conn"]
    assert fake_pool.closed is False

    saver.close()
    assert fake_pool.closed is True
