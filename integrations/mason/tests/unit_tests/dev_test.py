"""Unit tests for `mason dev`: wraps `databricks apps run-local` from the project dir."""

from __future__ import annotations

import pathlib
from unittest import mock

from click.testing import CliRunner

from databricks_mason import dev as dev_mod


class _Ctx:
    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile


def test_dev_runs_run_local_in_source_dir(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx(profile="ml")
        )
    assert result.exit_code == 0, result.output
    args, kwargs = db.call_args
    assert args[0][:2] == ["apps", "run-local"]
    assert "--prepare-environment" in args[0]  # on by default
    assert args[1] == "ml"  # profile passed through
    assert kwargs["cwd"] == str(tmp_path)  # runs in the project dir


def test_dev_no_prepare_and_custom_port(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev,
            ["--source", str(tmp_path), "--no-prepare-environment", "--app-port", "9000"],
            obj=_Ctx(),
        )
    assert result.exit_code == 0, result.output
    cmd = db.call_args.args[0]
    assert "--prepare-environment" not in cmd
    assert cmd[-2:] == ["--app-port", "9000"]


def test_dev_requires_app_yaml(tmp_path: pathlib.Path):
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert "app.yaml" in result.output
    db.assert_not_called()
