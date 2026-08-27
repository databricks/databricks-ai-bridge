"""Unit tests for `mason dev`."""

from types import SimpleNamespace
from unittest import mock

from click.testing import CliRunner

from databricks_mason.dev import dev


class _Ctx:
    profile = "demo-profile"


def test_dev_runs_project_start_script(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='agent'\nversion='0'\n")
    run = mock.Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr("databricks_mason.dev.subprocess.run", run)

    result = CliRunner().invoke(dev, ["--source", str(tmp_path), "--port", "9000"], obj=_Ctx())

    assert result.exit_code == 0, result.output
    args, kwargs = run.call_args
    assert args[0] == ["uv", "run", "start-server"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["DATABRICKS_CONFIG_PROFILE"] == "demo-profile"
    assert kwargs["env"]["PORT"] == "9000"


def test_dev_requires_agent_project(tmp_path):
    result = CliRunner().invoke(dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code != 0
    assert "No pyproject.toml" in result.output
