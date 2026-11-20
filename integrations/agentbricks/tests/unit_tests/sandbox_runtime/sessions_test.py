"""Session list projections retain activity time without allocating compute."""

from unittest.mock import MagicMock

from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.sandbox.control import ControlStore
from databricks_agentkit.sandbox.lakebox import Lakebox


def test_sessions_sort_by_activity_and_preserve_latest_invocation_status():
    registry = MagicMock(namespace="demo")
    control = ControlStore(registry, MagicMock(), runtime_store=InMemoryRuntimeStore())
    control.initialize()
    try:
        for session_id, timestamp, stage in [
            ("recent-activity", "2026-11-15T12:02:00+00:00", "Running agent"),
            ("recent-submission", "2026-11-15T12:01:00+00:00", "Installing dependencies"),
        ]:
            row = control.create_run("agent", session_id, "turn", {})
            attempt = control.call(control.runtime_store.claim(row["id"]))
            control.call(
                control.runtime_store.append_event(
                    row["id"],
                    attempt.attempt,
                    {"type": "sandbox.stage", "stage": stage, "at": timestamp},
                )
            )
        rows = control.sessions("agent")
        assert [row["session_id"] for row in rows] == ["recent-activity", "recent-submission"]
        assert rows[0]["updated_at"] == "2026-11-15T12:02:00+00:00"
        assert rows[0]["stage"] == "Running agent"
        assert rows[0]["status"] == "running"
        assert not control.sessions("another-agent")
    finally:
        control.close()


def test_bulk_sandbox_status_handles_pagination_and_missing_compute():
    lakebox = Lakebox.__new__(Lakebox)
    lakebox.api = MagicMock(
        side_effect=[
            {"sandboxes": [{"sandboxId": "one", "status": "Running"}], "nextPageToken": "page-2"},
            {"sandboxes": [{"sandboxId": "two", "status": "Stopped"}]},
        ]
    )
    assert lakebox.statuses({"one", "two", "gone"}) == {
        "one": "running",
        "two": "stopped",
        "gone": "missing",
    }
    assert lakebox.api.call_args.kwargs["query"]["page_token"] == "page-2"
    assert all(call.args[0] == "GET" for call in lakebox.api.call_args_list)
