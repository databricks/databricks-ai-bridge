"""CLI tests for `ab memory bind/unbind` and `ab sessions bind/unbind`.

Backfills coverage for the store-binding command handlers, which mutate agent.toml.
Uses a real temp AgentProject and rejects workspace calls: binding only edits agent.toml.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject
from databricks_mason.cli.memory import memory
from databricks_mason.cli.sessions import sessions
from databricks_mason.project_config import write_project_metadata


class _Ctx:
    def __init__(self, output="text"):
        self.output = output

    def client(self):
        raise AssertionError("Binding and unbinding stores must not contact the workspace")


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    project = tmp_path / "agent-langgraph"
    (project / "agent").mkdir(parents=True)
    write_project_metadata(project, framework="langgraph", template="agent-langgraph")
    AgentProject.create(project, framework="langgraph", server="mason").write()
    return project


@pytest.mark.parametrize("output", ["text", "json"])
def test_memory_bind_and_unbind(tmp_path, output):
    project = _project(tmp_path)
    result = CliRunner().invoke(
        memory, ["bind", "my-mem", "--source", str(project)], obj=_Ctx(output)
    )
    assert result.exit_code == 0, result.output
    if output == "json":
        assert json.loads(result.output) == {
            "memory_store": "my-mem",
            "manifest": str(project / "agent.toml"),
        }
    reloaded = AgentProject.load(project)
    assert reloaded.memory_store == "my-mem"
    assert reloaded.memory_store_id is None

    r2 = CliRunner().invoke(memory, ["unbind", "--source", str(project)], obj=_Ctx())
    assert r2.exit_code == 0, r2.output
    assert AgentProject.load(project).memory_store is None


def test_memory_bind_does_not_require_existing_store(tmp_path):
    project = _project(tmp_path)
    r = CliRunner().invoke(memory, ["bind", "ghost", "--source", str(project)], obj=_Ctx())
    assert r.exit_code == 0, r.output
    assert AgentProject.load(project).memory_store == "ghost"


def test_memory_bind_rejects_obsolete_create_stores_option(tmp_path):
    project = _project(tmp_path)
    result = CliRunner().invoke(
        memory, ["bind", "ghost", "--no-create-stores", "--source", str(project)], obj=_Ctx()
    )
    assert result.exit_code == 2
    assert "No such option '--no-create-stores'" in result.output
    assert AgentProject.load(project).memory_store is None


@pytest.mark.parametrize("output", ["text", "json"])
def test_sessions_bind_and_unbind(tmp_path, output):
    project = _project(tmp_path)
    result = CliRunner().invoke(
        sessions, ["bind", "my-sess", "--source", str(project)], obj=_Ctx(output)
    )
    assert result.exit_code == 0, result.output
    if output == "json":
        assert json.loads(result.output) == {
            "session_store": "my-sess",
            "manifest": str(project / "agent.toml"),
        }
    assert AgentProject.load(project).session_store == "my-sess"

    r2 = CliRunner().invoke(sessions, ["unbind", "--source", str(project)], obj=_Ctx())
    assert r2.exit_code == 0, r2.output
    assert AgentProject.load(project).session_store is None


def test_unbind_when_nothing_bound_is_graceful(tmp_path):
    project = _project(tmp_path)
    r = CliRunner().invoke(memory, ["unbind", "--source", str(project)], obj=_Ctx())
    assert r.exit_code == 0
    assert "No memory store binding" in r.output
    assert AgentProject.load(project).memory_store is None
