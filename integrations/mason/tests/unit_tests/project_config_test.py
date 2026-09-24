"""Project metadata shares framework parsing with the canonical manifest."""

import pathlib

import pytest
import tomli

from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import load_project_metadata, write_project_metadata
from databricks_mason.project_types import AgentFramework


@pytest.mark.parametrize("framework", list(AgentFramework))
def test_metadata_serializes_framework_value_and_loads_enum(
    tmp_path: pathlib.Path, framework: AgentFramework
):
    path = write_project_metadata(tmp_path, framework=framework, template="agent-template")
    assert tomli.loads(path.read_text())["framework"] == framework.value
    assert load_project_metadata(tmp_path).framework is framework


def test_metadata_override_conflict_uses_values_in_error(tmp_path: pathlib.Path):
    write_project_metadata(tmp_path, framework=AgentFramework.LANGGRAPH, template="agent-template")

    with pytest.raises(AgentCliError) as error:
        load_project_metadata(tmp_path, framework_override=AgentFramework.OPENAI)

    assert "Framework override 'openai' conflicts with 'langgraph'" in error.value.message
    assert "AgentFramework" not in error.value.message


def test_metadata_rejects_unsupported_framework_before_writing(tmp_path: pathlib.Path):
    with pytest.raises(AgentCliError, match="Unsupported Agent Bricks framework 'unsupported'"):
        write_project_metadata(tmp_path, framework="unsupported", template="agent-template")
    assert not (tmp_path / ".mason").exists()


@pytest.mark.parametrize(
    ("package", "framework"),
    [
        ("databricks-langchain", AgentFramework.LANGGRAPH),
        ("databricks-openai", AgentFramework.OPENAI),
    ],
)
def test_framework_inferred_from_dependencies_is_enum(
    tmp_path: pathlib.Path, package: str, framework: AgentFramework
):
    (tmp_path / "pyproject.toml").write_text(f'[project]\ndependencies = ["{package}>=0.1"]\n')
    assert load_project_metadata(tmp_path).framework is framework
