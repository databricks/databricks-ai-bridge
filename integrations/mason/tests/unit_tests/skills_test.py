"""Unit tests for Unity Catalog skill discovery and exact bindings."""

from __future__ import annotations

import importlib
import json
import pathlib
from typing import Any, cast

import pytest
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject


class _Client:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    def list_uc_skills(self, schema, page_size=None, page_token=None):
        self.calls.append((schema, page_size, page_token))
        return next(self.pages)


class _Context:
    def __init__(self, client=None, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


class _RepeatingTokenClient:
    def __init__(self):
        self.calls = []

    def list_uc_skills(self, schema, page_size=None, page_token=None):
        self.calls.append((schema, page_size, page_token))
        if len(self.calls) > 2:
            raise AssertionError("list followed a repeated pagination token")
        return {"skills": [], "next_page_token": "loop"}


def _skills_command():
    return importlib.import_module("databricks_mason.skills").skills


def _project(tmp_path: pathlib.Path, *, framework: str = "langgraph") -> pathlib.Path:
    project = tmp_path / "agent"
    AgentProject.create(project, framework=framework).write()
    return project


def _runner_with_separate_stderr() -> CliRunner:
    runner_type = cast(Any, CliRunner)
    try:
        return runner_type(mix_stderr=False)
    except TypeError:
        return CliRunner()


def test_list_json_normalizes_sorts_deduplicates_and_paginates():
    client = _Client(
        [
            {
                "skills": [
                    {
                        "name": "skills/catalog.schema.zeta",
                        "bundle_name": "zeta-bundle",
                        "description": "Close the quarter.",
                        "etag": "zeta-etag",
                        "comment": "Finance owned",
                        "effective_owner": "finance@example.com",
                        "id": "not-part-of-the-cli-contract",
                    },
                    {
                        "name": "skills/catalog.schema.alpha",
                        "description": "First skill",
                    },
                    {"description": "missing name"},
                ],
                "next_page_token": "page-2",
            },
            {
                "skills": [
                    {
                        "name": "skills/catalog.schema.alpha",
                        "description": "duplicate must not replace first",
                    },
                    {"name": "skills/catalog.schema.beta"},
                ]
            },
        ]
    )

    result = _runner_with_separate_stderr().invoke(
        _skills_command(),
        ["list", "--schema", "catalog.schema"],
        obj=_Context(client, "json"),
    )

    assert result.exit_code == 0, result.output
    assert client.calls == [
        ("catalog.schema", None, None),
        ("catalog.schema", None, "page-2"),
    ]
    assert json.loads(result.stdout) == {
        "schema_version": 1,
        "skills": [
            {"name": "catalog.schema.alpha", "description": "First skill"},
            {"name": "catalog.schema.beta"},
            {
                "name": "catalog.schema.zeta",
                "bundle_name": "zeta-bundle",
                "description": "Close the quarter.",
                "etag": "zeta-etag",
                "comment": "Finance owned",
                "effective_owner": "finance@example.com",
            },
        ],
    }


@pytest.mark.parametrize("response", [None, {"skills": {"name": "not-a-list"}}])
def test_list_rejects_malformed_api_response(response):
    result = CliRunner().invoke(
        _skills_command(),
        ["list", "--schema", "catalog.schema"],
        obj=_Context(_Client([response])),
    )

    assert result.exit_code != 0
    assert "Skills API returned an invalid response" in result.output


def test_list_rejects_wrong_typed_pagination_token():
    result = _runner_with_separate_stderr().invoke(
        _skills_command(),
        ["list", "--schema", "catalog.schema"],
        obj=_Context(_Client([{"skills": [], "next_page_token": 42}]), "json"),
    )

    assert result.exit_code != 0
    assert "pagination token" in result.stderr.lower()
    assert "invalid response" in result.stderr.lower()


def test_list_rejects_repeated_pagination_token_without_requesting_it_again():
    client = _RepeatingTokenClient()

    result = _runner_with_separate_stderr().invoke(
        _skills_command(),
        ["list", "--schema", "catalog.schema"],
        obj=_Context(client, "json"),
    )

    assert result.exit_code != 0
    assert "repeated pagination token" in result.stderr.lower()
    assert client.calls == [
        ("catalog.schema", None, None),
        ("catalog.schema", None, "loop"),
    ]


def test_list_skips_malformed_entries_with_stderr_diagnostics_and_valid_json():
    client = _Client(
        [
            {
                "skills": [
                    {"name": "skills/catalog.schema.valid-skill", "description": "Valid."},
                    {"name": "garbage"},
                    {"name": "skills/not.three-part"},
                    {},
                    {"name": 42},
                    "not-an-object",
                ]
            }
        ]
    )

    result = _runner_with_separate_stderr().invoke(
        _skills_command(),
        ["list", "--schema", "catalog.schema"],
        obj=_Context(client, "json"),
    )

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "schema_version": 1,
        "skills": [{"name": "catalog.schema.valid-skill", "description": "Valid."}],
    }
    assert result.stderr.count("Warning: skipped malformed UC skill entry") == 5


def test_list_rejects_invalid_schema_before_api_call():
    client = _Client([])

    result = CliRunner().invoke(
        _skills_command(),
        ["list", "--schema", "catalog"],
        obj=_Context(client),
    )

    assert result.exit_code != 0
    assert "catalog.schema" in result.output
    assert client.calls == []


def test_list_text_shows_copyable_add_commands():
    client = _Client(
        [
            {
                "skills": [
                    {"name": "skills/catalog.schema.close-quarter"},
                    {"name": "skills/catalog.schema.audit-ledger"},
                ]
            }
        ]
    )

    result = CliRunner().invoke(
        _skills_command(),
        ["list", "--schema", "catalog.schema"],
        obj=_Context(client),
    )

    assert result.exit_code == 0, result.output
    assert "mason skills add uc catalog.schema.audit-ledger" in result.output
    assert "mason skills add uc catalog.schema.close-quarter" in result.output


def test_add_uc_is_exact_and_idempotent(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    args = [
        "add",
        "uc",
        "catalog.schema.skill",
        "--name",
        "review",
        "--source",
        str(project),
    ]
    runner = CliRunner()

    first = runner.invoke(_skills_command(), args, obj=_Context(output="json"))
    second = runner.invoke(_skills_command(), args, obj=_Context(output="json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output) == {
        "schema_version": 1,
        "changed": True,
        "changed_files": [str(project / "agent.toml")],
        "skill": {
            "id": "review",
            "kind": "uc",
            "source": "catalog.schema.skill",
        },
    }
    assert json.loads(second.output)["changed"] is False
    assert json.loads(second.output)["changed_files"] == []
    assert len(AgentProject.load(project).skills) == 1


def test_add_uc_defaults_id_to_fqn_leaf(tmp_path: pathlib.Path):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        _skills_command(),
        ["add", "uc", "catalog.schema.skill", "--source", str(project)],
        obj=_Context(),
    )

    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).skills[0].id == "skill"


def test_add_exposes_only_uc_sources():
    add = _skills_command().commands["add"]

    assert set(add.commands) == {"uc"}


def test_add_uc_rejects_openai_project(tmp_path: pathlib.Path):
    project = _project(tmp_path, framework="openai")

    result = CliRunner().invoke(
        _skills_command(),
        ["add", "uc", "catalog.schema.skill", "--source", str(project)],
        obj=_Context(),
    )

    assert result.exit_code != 0
    assert "Agent skills are LangGraph-only" in result.output
    assert AgentProject.load(project).skills == []
