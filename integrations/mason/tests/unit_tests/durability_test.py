"""Unit tests for `mason durability bind|unbind`."""

from __future__ import annotations

from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject
from databricks_mason.durability import durability


class _Ctx:
    def __init__(self, output: str = "text"):
        self.output = output


def _write_manifest(root) -> None:
    AgentProject.create(root, framework="langgraph").write()


def test_durability_bind_writes_agent_toml(tmp_path) -> None:
    _write_manifest(tmp_path)

    result = CliRunner().invoke(
        durability,
        ["bind", "--source", str(tmp_path)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert AgentProject.load(tmp_path).durability_enabled is True
    assert "Enabled durable invocation storage" in result.output


def test_durability_bind_is_idempotent_and_supports_json(tmp_path) -> None:
    _write_manifest(tmp_path)
    runner = CliRunner()

    first = runner.invoke(durability, ["bind", "--source", str(tmp_path)], obj=_Ctx("json"))
    second = runner.invoke(durability, ["bind", "--source", str(tmp_path)], obj=_Ctx("json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert '"durability": true' in second.output


def test_durability_unbind_clears_agent_toml(tmp_path) -> None:
    project = AgentProject.create(tmp_path, framework="langgraph")
    project.bind_durability()
    project.write()

    result = CliRunner().invoke(
        durability,
        ["unbind", "--source", str(tmp_path)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert AgentProject.load(tmp_path).durability_enabled is False
    assert "Disabled durable invocation storage" in result.output
