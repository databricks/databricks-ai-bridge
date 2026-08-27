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


def test_dev_prepares_when_no_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")  # no .venv -> auto-prepare
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx(profile="ml")
        )
    assert result.exit_code == 0, result.output
    args, kwargs = db.call_args
    assert args[0][:2] == ["apps", "run-local"]
    assert "--prepare-environment" in args[0]  # no venv yet -> build it
    assert args[1] == "ml"  # profile passed through
    assert kwargs["cwd"] == str(tmp_path)  # runs in the project dir


def test_dev_reuses_existing_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()  # env already there -> don't rebuild
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "--prepare-environment" not in db.call_args.args[0]


def test_dev_force_prepare_overrides_existing_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path), "--prepare-environment"], obj=_Ctx()
        )
    assert result.exit_code == 0, result.output
    assert "--prepare-environment" in db.call_args.args[0]  # explicit flag forces rebuild


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
