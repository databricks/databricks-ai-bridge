"""Unit tests for the canonical ``agent.toml`` project model."""

from __future__ import annotations

import pathlib

import pytest

from databricks_mason import agent_project as project_model
from databricks_mason.agent_project import AgentProject, Scope, ToolSpec
from databricks_mason.errors import AgentCliError


def _write_manifest(root: pathlib.Path, body: str | None = None) -> pathlib.Path:
    path = root / "agent.toml"
    path.write_text(
        body or 'schema_version = 1\n# keep me\n\n[agent]\nframework = "langgraph"\n',
        encoding="utf-8",
    )
    return path


def test_agent_project_round_trips_tool_specs_without_losing_comments(tmp_path: pathlib.Path):
    path = _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)

    changed = project.add_tool(
        ToolSpec.sandbox("sandbox", scopes=[Scope.table("samples.nyctaxi.trips")])
    )
    project.write()

    assert changed is True
    assert "# keep me" in path.read_text(encoding="utf-8")
    loaded = AgentProject.load(tmp_path)
    assert loaded.framework == "langgraph"
    assert loaded.tools[0].source.kind == "sandbox"
    assert loaded.tools[0].policy.downscope == (
        Scope(kind="table", value="samples.nyctaxi.trips", permission="read_only"),
    )


def test_add_same_tool_is_idempotent(tmp_path: pathlib.Path):
    _write_manifest(tmp_path, 'schema_version = 1\n\n[agent]\nframework = "openai"\n')
    project = AgentProject.load(tmp_path)
    spec = ToolSpec.mcp("web", service="system.ai.web_search")

    assert project.add_tool(spec) is True
    assert project.add_tool(spec) is False


def test_add_conflicting_tool_id_fails_without_writing(tmp_path: pathlib.Path):
    path = _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    project.add_tool(ToolSpec.mcp("shared", service="system.ai.web_search"))
    project.write()
    before = path.read_text(encoding="utf-8")

    with pytest.raises(AgentCliError, match="already exists"):
        project.add_tool(ToolSpec.uc_function("shared", function="main.tools.lookup"))

    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ToolSpec.mcp("web", service="not-three-parts"), "MCP service"),
        (lambda: ToolSpec.uc_function("lookup", function="catalog.schema"), "UC function"),
        (lambda: ToolSpec.sandbox("sandbox", scopes=[]), "scope"),
        (
            lambda: ToolSpec.sandbox("sandbox", scopes=[Scope(kind="unknown", value="c.s.t")]),
            "scope kind",
        ),
    ],
)
def test_tool_spec_rejects_invalid_resources(factory, message: str):
    with pytest.raises(AgentCliError, match=message):
        factory()


def test_load_rejects_unsupported_schema_before_mutation(tmp_path: pathlib.Path):
    path = _write_manifest(tmp_path, 'schema_version = 2\n\n[agent]\nframework = "openai"\n')
    before = path.read_text(encoding="utf-8")

    with pytest.raises(AgentCliError, match="schema"):
        AgentProject.load(tmp_path)

    assert path.read_text(encoding="utf-8") == before


def test_write_is_atomic_when_replace_fails(tmp_path: pathlib.Path, monkeypatch):
    path = _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    project.add_tool(ToolSpec.mcp("web", service="system.ai.web_search"))
    before = path.read_text(encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr("databricks_mason.agent_project.os.replace", fail_replace)
    with pytest.raises(AgentCliError, match="replace failed"):
        project.write()

    assert path.read_text(encoding="utf-8") == before


def test_uc_skill_round_trips_and_add_is_idempotent(tmp_path: pathlib.Path):
    project = AgentProject.create(tmp_path, framework="langgraph")
    spec = project_model.SkillSpec.uc("review", name="catalog.schema.skill")

    assert project.add_skill(spec) is True
    assert project.add_skill(spec) is False
    project.write()

    assert "[[skills]]" in (tmp_path / "agent.toml").read_text(encoding="utf-8")
    assert AgentProject.load(tmp_path).skills == [spec]


def test_uc_skill_rejects_conflicting_id(tmp_path: pathlib.Path):
    project = AgentProject.create(tmp_path, framework="langgraph")
    project.add_skill(project_model.SkillSpec.uc("review", name="main.ai.review"))

    with pytest.raises(AgentCliError, match="Skill id 'review' already exists"):
        project.add_skill(project_model.SkillSpec.uc("review", name="main.ai.other-review"))


@pytest.mark.parametrize(
    "name",
    [
        "catalog.schema",
        "catalog..review",
        "catalog.schema.review.extra",
        "catalog.schema.project review",
    ],
)
def test_uc_skill_rejects_invalid_three_part_name(name: str):
    with pytest.raises(AgentCliError, match="UC skill"):
        project_model.SkillSpec.uc("review", name=name)


def test_load_rejects_duplicate_skill_ids(tmp_path: pathlib.Path):
    _write_manifest(
        tmp_path,
        """schema_version = 1

[agent]
framework = "langgraph"

[[skills]]
id = "review"
source = { kind = "uc", name = "main.ai.review" }

[[skills]]
id = "review"
source = { kind = "uc", name = "main.ai.other-review" }
""",
    )

    with pytest.raises(AgentCliError, match="skill ids must be unique"):
        AgentProject.load(tmp_path)


@pytest.mark.parametrize(
    "skill_entry",
    [
        'id = "review"',
        'id = "review"\nsource = "uc"',
        'id = "review"\nsource = { kind = "unknown" }',
        'id = "review"\nsource = { kind = "local", path = "skills/review" }',
        'id = "review"\nsource = { kind = "uc" }',
        'id = "review"\nsource = { kind = "uc", name = 42 }',
        (
            'id = "review"\n'
            'source = { kind = "uc", name = "main.ai.review", path = "skills/review" }'
        ),
        'id = "review"\nsource = { kind = "uc", name = "main.ai.review" }\nversion = "v1"',
    ],
)
def test_load_rejects_malformed_or_non_uc_skill_sources(tmp_path: pathlib.Path, skill_entry: str):
    _write_manifest(
        tmp_path,
        f"""schema_version = 1

[agent]
framework = "langgraph"

[[skills]]
{skill_entry}
""",
    )

    with pytest.raises(AgentCliError, match="[Ss]kill|source"):
        AgentProject.load(tmp_path)


def test_openai_project_rejects_skill_addition(tmp_path: pathlib.Path):
    project = AgentProject.create(tmp_path, framework="openai")

    with pytest.raises(AgentCliError, match="Agent skills are LangGraph-only"):
        project.add_skill(project_model.SkillSpec.uc("review", name="main.ai.review"))

    assert project.skills == []


def test_load_rejects_openai_manifest_with_skills(tmp_path: pathlib.Path):
    path = _write_manifest(
        tmp_path,
        """schema_version = 1

[agent]
framework = "openai"

[[skills]]
id = "review"
source = { kind = "uc", name = "main.ai.review" }
""",
    )
    before = path.read_text(encoding="utf-8")

    with pytest.raises(AgentCliError, match="Agent skills are LangGraph-only"):
        AgentProject.load(tmp_path)

    assert path.read_text(encoding="utf-8") == before
