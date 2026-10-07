"""Behavior tests for ``agentbricks memory pipeline`` memory-pipeline management."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import databricks_agentbricks.cli.app as cli

PIPELINE = {
    "name": "memory-pipelines/p-123",
    "display_name": "support-pipeline",
    "session_store": "session-stores/support-sessions",
    "memory_store": "memory-stores/support-memory",
    "model": "system.ai.gpt-5-6-sol",
    "dreamer_policy": {
        "enabled": True,
        "trigger": "MANUAL_ONLY",
        "instructions": "Keep durable customer preferences.",
    },
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


def _pipeline():
    assert "pipeline" in cli.memory.commands, "memory must register the pipeline command group"
    assert "dreamer" not in cli.memory.commands
    return cli.memory.commands["pipeline"]


def test_create_accepts_store_names_and_model():
    client = _Client()
    result = CliRunner().invoke(
        _pipeline(),
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
                "trigger": "manual",
            },
        )
    ]
    assert "memory-pipelines/p-123" in result.output


def test_create_with_scheduled_trigger():
    client = _Client()
    result = CliRunner().invoke(
        _pipeline(),
        ["create", "--memory-store", "m", "--session-store", "s", "--trigger", "scheduled"],
        obj=_Ctx(client),
    )

    assert result.exit_code == 0, result.output
    assert client.calls[0][1]["trigger"] == "scheduled"


def test_create_rejects_unknown_trigger_before_api_call():
    client = _Client()
    result = CliRunner().invoke(
        _pipeline(),
        ["create", "--memory-store", "m", "--session-store", "s", "--trigger", "hourly"],
        obj=_Ctx(client),
    )

    assert result.exit_code == 2
    assert "Invalid value for '--trigger'" in result.output
    assert client.calls == []


def test_list_get_update_and_delete_expose_crud_workflow():
    client = _Client()
    ctx = _Ctx(client)
    runner = CliRunner()
    pipeline = _pipeline()

    listed = runner.invoke(pipeline, ["list", "--page-size", "10"], obj=ctx)
    fetched = runner.invoke(pipeline, ["get", "p-123"], obj=ctx)
    updated = runner.invoke(
        pipeline,
        ["update", "p-123", "--instructions", "Only durable facts."],
        obj=ctx,
    )
    deleted = runner.invoke(pipeline, ["delete", "p-123", "--yes"], obj=ctx)

    for result in (listed, fetched, updated, deleted):
        assert result.exit_code == 0, result.output
    assert "Keep durable customer preferences." in fetched.output
    assert client.calls == [
        ("list", 10, None),
        ("get", "p-123"),
        (
            "update",
            "p-123",
            {"display_name": None, "instructions": "Only durable facts."},
        ),
        ("delete", "p-123"),
    ]


def test_run_triggers_pipeline_and_renders_returned_run():
    client = _Client()
    result = CliRunner().invoke(_pipeline(), ["run", "p-123"], obj=_Ctx(client))

    assert result.exit_code == 0, result.output
    assert client.calls == [("run", "p-123")]
    assert "memory-pipelines/p-123/runs/run-456" in result.output
    assert "PIPELINE_RUN_STATE_PENDING" in result.output


@pytest.mark.parametrize("command", ["create", "update"])
def test_instructions_from_file(command, tmp_path):
    instructions = "# Distillation\n\nKeep durable preferences — including context.\n"
    path = tmp_path / "instructions.md"
    path.write_text(instructions, encoding="utf-8")
    client = _Client()
    args = (
        ["create", "--memory-store", "m", "--session-store", "s"]
        if command == "create"
        else ["update", "p-123"]
    )
    result = CliRunner().invoke(
        _pipeline(), [*args, "--instructions", f"@{path}"], obj=_Ctx(client)
    )

    assert result.exit_code == 0, result.output
    assert client.calls[0][-1]["instructions"] == instructions


@pytest.mark.parametrize("command", ["create", "update"])
def test_missing_instructions_file_fails_before_api_call(command, tmp_path):
    client = _Client()
    args = (
        ["create", "--memory-store", "m", "--session-store", "s"]
        if command == "create"
        else ["update", "p-123"]
    )
    result = CliRunner().invoke(
        _pipeline(),
        [*args, "--instructions", f"@{tmp_path / 'missing.md'}"],
        obj=_Ctx(client),
    )

    assert result.exit_code == 2
    assert "Invalid value for '--instructions'" in result.output
    assert client.calls == []
