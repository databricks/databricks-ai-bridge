"""Behavior tests for ``ab dreamer`` memory-pipeline management."""

from __future__ import annotations

from click.testing import CliRunner

import databricks_agentbricks.cli.app as cli

PIPELINE = {
    "name": "memory-pipelines/p-123",
    "display_name": "support-dreamer",
    "session_store": "session-stores/support-sessions",
    "memory_store": "memory-stores/support-memory",
    "instructions": "Keep durable customer preferences.",
    "model": "system.ai.gpt-5-6-sol",
    "dreamer_policy": {"enabled": True, "trigger": "MANUAL_ONLY"},
    "etag": "etag-1",
    "create_time": "2026-09-24T01:00:00Z",
    "update_time": "2026-09-24T02:00:00Z",
}


class _Client:
    def __init__(self):
        self.calls = []

    def create_memory_pipeline(self, **kwargs):
        self.calls.append(("create", kwargs))
        return PIPELINE

    def list_memory_pipelines(self, page_size=None, page_token=None):
        self.calls.append(("list", page_size, page_token))
        return {"memory_pipelines": [PIPELINE]}

    def get_memory_pipeline(self, name):
        self.calls.append(("get", name))
        return PIPELINE

    def update_memory_pipeline(self, name, **kwargs):
        self.calls.append(("update", name, kwargs))
        return {**PIPELINE, **kwargs}

    def delete_memory_pipeline(self, name):
        self.calls.append(("delete", name))
        return {}

    def run_memory_pipeline(self, name):
        self.calls.append(("run", name))
        return {
            "name": f"memory-pipelines/{name}/runs/run-456",
            "state": "PIPELINE_RUN_STATE_PENDING",
            "create_time": "2026-09-24T03:00:00Z",
        }


class _Ctx:
    output = "text"

    def __init__(self, client):
        self._client = client

    def client(self):
        return self._client


def _dreamer():
    assert "dreamer" in cli.agentbricks.commands, "ab must register the dreamer command group"
    return cli.agentbricks.commands["dreamer"]


def test_create_accepts_store_names_and_model():
    client = _Client()
    result = CliRunner().invoke(
        _dreamer(),
        [
            "create",
            "--memory-store",
            "support-memory",
            "--session-store",
            "support-sessions",
            "--model",
            "system.ai.gpt-5-6-sol",
        ],
        obj=_Ctx(client),
    )

    assert result.exit_code == 0, result.output
    assert client.calls == [
        (
            "create",
            {
                "memory_store": "support-memory",
                "session_store": "support-sessions",
                "model": "system.ai.gpt-5-6-sol",
                "display_name": None,
                "instructions": None,
            },
        )
    ]
    assert "memory-pipelines/p-123" in result.output


def test_list_get_update_and_delete_expose_crud_workflow():
    client = _Client()
    ctx = _Ctx(client)
    runner = CliRunner()
    dreamer = _dreamer()

    listed = runner.invoke(dreamer, ["list", "--page-size", "10"], obj=ctx)
    fetched = runner.invoke(dreamer, ["get", "p-123"], obj=ctx)
    updated = runner.invoke(
        dreamer,
        ["update", "p-123", "--instructions", "Only durable facts.", "--disable"],
        obj=ctx,
    )
    deleted = runner.invoke(dreamer, ["delete", "p-123", "--yes"], obj=ctx)

    for result in (listed, fetched, updated, deleted):
        assert result.exit_code == 0, result.output
    assert client.calls == [
        ("list", 10, None),
        ("get", "p-123"),
        (
            "update",
            "p-123",
            {"display_name": None, "instructions": "Only durable facts.", "enabled": False},
        ),
        ("delete", "p-123"),
    ]


def test_run_triggers_pipeline_and_renders_returned_run():
    client = _Client()
    result = CliRunner().invoke(_dreamer(), ["run", "p-123"], obj=_Ctx(client))

    assert result.exit_code == 0, result.output
    assert client.calls == [("run", "p-123")]
    assert "memory-pipelines/p-123/runs/run-456" in result.output
    assert "PIPELINE_RUN_STATE_PENDING" in result.output
