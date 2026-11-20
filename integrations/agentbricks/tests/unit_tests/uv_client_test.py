"""Unit coverage for the ``uv sync`` step behind ``agentbricks dev``."""

from __future__ import annotations

import pathlib
import subprocess
from unittest import mock

import pytest

from databricks_agentbricks.clients import uv as uv_module
from databricks_agentbricks.errors import AgentCliError


def test_uv_sync_runs_in_the_project_without_an_activated_venv(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/elsewhere/.venv")
    with mock.patch.object(
        uv_module.subprocess, "run", return_value=subprocess.CompletedProcess(["uv"], 0)
    ) as run:
        uv_module.uv_sync(tmp_path)

    args, kwargs = run.call_args
    assert args[0] == ["uv", "sync"]
    assert kwargs["cwd"] == tmp_path
    assert "VIRTUAL_ENV" not in kwargs["env"]


def test_uv_sync_failure_is_a_cli_error(tmp_path: pathlib.Path):
    with mock.patch.object(
        uv_module.subprocess, "run", return_value=subprocess.CompletedProcess(["uv"], 1)
    ):
        with pytest.raises(AgentCliError, match="Could not install the agent's dependencies"):
            uv_module.uv_sync(tmp_path)


def test_uv_sync_without_uv_says_how_to_install_it(tmp_path: pathlib.Path):
    with mock.patch.object(uv_module.subprocess, "run", side_effect=FileNotFoundError("uv")):
        with pytest.raises(AgentCliError, match="Could not find `uv`") as error:
            uv_module.uv_sync(tmp_path)
    assert error.value.hint is not None and "astral" in error.value.hint
