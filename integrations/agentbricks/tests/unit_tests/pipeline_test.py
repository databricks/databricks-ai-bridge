"""Behavior tests for ``agentbricks memory pipeline`` memory-pipeline management."""

from __future__ import annotations

import json
from unittest import mock

import pytest
from click.testing import CliRunner

import databricks_agentbricks.cli.app as cli
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit._api_client import _AgentBricksApiClient

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
    profile = None

    def __init__(self, client):
        self._client = client

    def client(self):
        return self._client


def _pipeline():
    assert "pipeline" in cli.memory.commands, "memory must register the pipeline command group"
    assert "dreamer" not in cli.memory.commands
    return cli.memory.commands["pipeline"]


@pytest.mark.parametrize("output", ["text", "json"])
def test_get_run_uses_pipeline_api(output):
    workspace = mock.Mock()
    response = {
        "name": "memory-pipelines/p-123/runs/418898433707211",
        "state": "SUCCEEDED",
        "create_time": "2026-10-16T01:00:00Z",
    }
    workspace.api_client.do.return_value = response
    ctx = _Ctx(_AgentBricksApiClient(workspace_client=workspace))
    ctx.output = output
    result = CliRunner().invoke(
        _pipeline(), ["runs", "get", "memory-pipelines/p-123/runs/418898433707211"], obj=ctx
    )
    assert result.exit_code == 0, result.output
    workspace.api_client.do.assert_called_once_with(
        "GET",
        "/api/2.0/agents/memory-pipelines/p-123/runs/418898433707211",
        query=None,
        body=None,
    )
    if output == "json":
        assert json.loads(result.stdout) == response
    else:
        assert response["name"] in result.stdout
        assert "SUCCEEDED" in result.stdout


def test_get_run_surfaces_api_error():
    workspace = mock.Mock()
    workspace.api_client.do.side_effect = AgentCliError("Run not found", error_code="NOT_FOUND")
    result = CliRunner().invoke(
        _pipeline(),
        ["runs", "get", "memory-pipelines/p-123/runs/123"],
        obj=_Ctx(_AgentBricksApiClient(workspace_client=workspace)),
    )
    assert result.exit_code == 1
    assert "Run not found" in result.output


@pytest.mark.parametrize(
    "name", ["123", "memory-pipelines/p/runs/abc", "memory-pipelines/p/runs/123/extra"]
)
def test_get_run_rejects_invalid_resource_name(name):
    result = CliRunner().invoke(_pipeline(), ["runs", "get", name], obj=_Ctx(_Client()))
    assert result.exit_code == 2


@pytest.mark.parametrize("command", [["get", "p-123"], ["list"]])
def test_pipeline_renders_nested_model(command, monkeypatch):
    policy = PIPELINE["dreamer_policy"]
    assert isinstance(policy, dict)
    monkeypatch.setitem(PIPELINE, "dreamer_policy", {**policy, "model": "nested-model"})
    result = CliRunner().invoke(_pipeline(), command, obj=_Ctx(_Client()))
    assert result.exit_code == 0, result.output
    assert "nested-model" in result.output
    assert "system.ai.gpt-5-6-sol" not in result.output


@pytest.mark.parametrize("command,trigger", [("pause", "MANUAL_ONLY"), ("resume", "SCHEDULED")])
@pytest.mark.parametrize("output", ["text", "json"])
@pytest.mark.parametrize("name", ["p-123", "memory-pipelines/p-123"])
def test_pause_resume_update_only_trigger(command, trigger, output, name):
    workspace = mock.Mock()
    policy = PIPELINE["dreamer_policy"]
    assert isinstance(policy, dict)
    response = {
        **PIPELINE,
        "dreamer_policy": {**policy, "trigger": trigger},
    }
    previous_trigger = "SCHEDULED" if command == "pause" else "MANUAL_ONLY"
    workspace.api_client.do.side_effect = [
        {**PIPELINE, "dreamer_policy": {**policy, "trigger": previous_trigger}},
        response,
    ]
    ctx = _Ctx(_AgentBricksApiClient(workspace_client=workspace))
    ctx.output = output
    result = CliRunner().invoke(_pipeline(), [command, name], obj=ctx)
    assert result.exit_code == 0, result.output
    assert workspace.api_client.do.call_args_list == [
        mock.call("GET", "/api/2.0/agents/memory-pipelines/p-123", query=None, body=None),
        mock.call(
            "PATCH",
            "/api/2.0/agents/memory-pipelines/p-123",
            query={"update_mask": "dreamer_policy.trigger"},
            body={"name": "memory-pipelines/p-123", "dreamer_policy": {"trigger": trigger}},
        ),
    ]
    if output == "json":
        assert json.loads(result.stdout) == response
    else:
        assert trigger in result.stdout


@pytest.mark.parametrize(
    "command,policy,state",
    [
        ("pause", {"trigger": "MANUAL_ONLY"}, "paused"),
        ("pause", {"trigger": "TRIGGER_UNSPECIFIED"}, "paused"),
        ("pause", {}, "paused"),
        ("resume", {"trigger": "SCHEDULED"}, "resumed"),
    ],
)
def test_pause_resume_rejects_existing_state(command, policy, state):
    workspace = mock.Mock()
    workspace.api_client.do.return_value = {"dreamer_policy": policy}
    ctx = _Ctx(_AgentBricksApiClient(workspace_client=workspace))
    result = CliRunner().invoke(_pipeline(), [command, "p-123"], obj=ctx)
    assert result.exit_code != 0
    assert f"already {state}" in result.output
    workspace.api_client.do.assert_called_once_with(
        "GET", "/api/2.0/agents/memory-pipelines/p-123", query=None, body=None
    )


@pytest.mark.parametrize("name", ["p-123", "memory-pipelines/p-123"])
@pytest.mark.parametrize("output", ["text", "json"])
def test_list_runs_calls_api_and_preserves_pagination(name, output):
    workspace = mock.Mock()
    response = {
        "pipeline_runs": [
            {
                "name": "memory-pipelines/p-123/runs/456",
                "state": "SUCCEEDED",
                "create_time": "2026-10-15T01:00:00Z",
            }
        ],
        "next_page_token": "next-page",
    }
    workspace.api_client.do.return_value = response
    ctx = _Ctx(_AgentBricksApiClient(workspace_client=workspace))
    ctx.output = output
    result = CliRunner().invoke(
        _pipeline(),
        ["runs", "list", name, "--page-size", "10", "--page-token", "previous-page"],
        obj=ctx,
    )
    assert result.exit_code == 0, result.output
    workspace.api_client.do.assert_called_once_with(
        "GET",
        "/api/2.0/agents/memory-pipelines/p-123/runs",
        query={"page_size": 10, "page_token": "previous-page"},
        body=None,
    )
    if output == "json":
        assert json.loads(result.stdout) == response
    else:
        assert "memory-pipelines/p-123/runs/456" in result.stdout
        assert "SUCCEEDED" in result.stdout
        assert "--page-token next-page" in result.stdout


@pytest.mark.parametrize("response", [{}, {"pipeline_runs": []}])
def test_list_runs_empty_response_and_default_page_size(response):
    workspace = mock.Mock()
    workspace.api_client.do.return_value = response
    result = CliRunner().invoke(
        _pipeline(),
        ["runs", "list", "p-123"],
        obj=_Ctx(_AgentBricksApiClient(workspace_client=workspace)),
    )
    assert result.exit_code == 0, result.output
    workspace.api_client.do.assert_called_once_with(
        "GET", "/api/2.0/agents/memory-pipelines/p-123/runs", query={"page_size": 50}, body=None
    )
    assert "Next page" not in result.stdout


def test_list_runs_rejects_negative_page_size():
    workspace = mock.Mock()
    result = CliRunner().invoke(
        _pipeline(),
        ["runs", "list", "p-123", "--page-size", "-1"],
        obj=_Ctx(_AgentBricksApiClient(workspace_client=workspace)),
    )
    assert result.exit_code == 2
    workspace.api_client.do.assert_not_called()


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
            {
                "display_name": None,
                "instructions": "Only durable facts.",
                "model": None,
                "trigger": None,
            },
        ),
        ("delete", "p-123"),
    ]


def test_list_shows_store_ids_without_collection_prefix():
    result = CliRunner().invoke(_pipeline(), ["list"], obj=_Ctx(_Client()))

    assert result.exit_code == 0, result.output
    # Full resource names get cut off in the table, hiding the part that identifies the store.
    assert "support-sessions" in result.output
    assert "support-memory" in result.output
    assert "session-stores/" not in result.output
    assert "memory-stores/" not in result.output


@pytest.mark.parametrize("model", ["system.ai.gpt-5-6-sol", ""])
def test_update_accepts_model_and_instructions_file(model, tmp_path):
    path = tmp_path / "instructions.md"
    instructions = "# Distillation\nKeep durable facts.\n"
    path.write_text(instructions, encoding="utf-8")
    client = _Client()

    result = CliRunner().invoke(
        _pipeline(),
        ["update", "p-123", "--model", model, "--instructions", f"@{path}"],
        obj=_Ctx(client),
    )

    assert result.exit_code == 0, result.output
    assert client.calls == [
        (
            "update",
            "p-123",
            {"display_name": None, "instructions": instructions, "model": model, "trigger": None},
        )
    ]


@pytest.mark.parametrize("trigger", ["manual", "scheduled"])
def test_update_accepts_trigger(trigger):
    client = _Client()
    result = CliRunner().invoke(
        _pipeline(), ["update", "p-123", "--trigger", trigger], obj=_Ctx(client)
    )
    assert result.exit_code == 0, result.output
    assert client.calls[0][-1]["trigger"] == trigger


def test_update_rejects_unknown_trigger():
    client = _Client()
    result = CliRunner().invoke(
        _pipeline(), ["update", "p-123", "--trigger", "hourly"], obj=_Ctx(client)
    )
    assert result.exit_code == 2
    assert client.calls == []


def test_run_triggers_pipeline_and_renders_returned_run():
    client = _Client()
    result = CliRunner().invoke(_pipeline(), ["run", "p-123"], obj=_Ctx(client))

    assert result.exit_code == 0, result.output
    assert client.calls == [("run", "p-123")]
    assert "memory-pipelines/p-123/runs/run-456" in result.output
    assert "PIPELINE_RUN_STATE_PENDING" in result.output


def test_run_conflict_points_at_in_progress_run():
    active = "memory-pipelines/p-123/runs/501787146174076"

    class ConflictClient(_Client):
        def run_memory_pipeline(self, name):
            raise AgentCliError(
                f"memory-pipelines/{name} already has a run in progress: {active}. "
                "Wait for it to finish before starting another.",
                error_code="ABORTED",
            )

    result = CliRunner().invoke(
        _pipeline(), ["run", "p-123"], obj=_Ctx(ConflictClient()), standalone_mode=False
    )

    assert isinstance(result.exception, AgentCliError)
    assert result.exception.error_code == "ABORTED"
    # The user can't fix a conflict by re-running; tell them how to inspect the blocking run.
    assert result.exception.hint == (
        f"Check its progress with `agentbricks memory pipeline runs get {active}`."
    )


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
