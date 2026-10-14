"""Shared fixtures for the Agent Bricks CLI unit tests.

Service tests inject their collaborators directly; a directory-wide patch of the old deploy
module would mask the new boundaries and fails because that function no longer exists.

Every test runs against an empty Databricks config file and none of the ambient credential or CI
variables, so profile resolution never depends on the developer's machine or the CI runner.
"""

from __future__ import annotations

import pathlib
import textwrap
from collections.abc import Callable
from unittest import mock

import pytest

from databricks_agentbricks.agent_hint import AGENT_ENV_MARKERS, AGENT_ENV_VALUES
from databricks_agentbricks.cli import auth
from databricks_agentbricks.clients import databricks_cli
from databricks_agentbricks.services import workspace_auth

_AMBIENT_ENV = ("DATABRICKS_CONFIG_PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN", "CI")


@pytest.fixture(autouse=True)
def databrickscfg_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """Isolated (initially absent) `DATABRICKS_CONFIG_FILE` and `HOME`; the terminal is non-interactive.

    HOME points at a tmp dir so nothing reads the developer's real config or token cache.
    """
    for name in _AMBIENT_ENV:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path_factory.mktemp("databricks-config") / ".databrickscfg"
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(path))
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("agentbricks-home")))
    monkeypatch.setattr(auth, "_is_interactive", lambda: False)
    return path


@pytest.fixture
def write_databrickscfg(databrickscfg_path: pathlib.Path) -> Callable[[str], None]:
    """Write the isolated Databricks config file from an INI snippet."""

    def _write(text: str) -> None:
        databrickscfg_path.write_text(textwrap.dedent(text))

    return _write


@pytest.fixture
def validated_profile(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Make the bounded SDK validation (the workspace boundary) succeed for any profile."""
    client = mock.Mock(host="https://ws.example", current_user="me@example.com")
    validate = mock.Mock(return_value=((client, "me@example.com"), None))
    monkeypatch.setattr(workspace_auth, "_validate_bounded", validate)
    return validate


@pytest.fixture
def interactive_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Someone is at the terminal, so a browser login could be completed."""
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)


@pytest.fixture
def fail_validation(monkeypatch: pytest.MonkeyPatch) -> Callable[[Exception], None]:
    """Make the bounded SDK validation fail with the given error for any profile."""

    def _fail(error: Exception) -> None:
        monkeypatch.setattr(workspace_auth, "_validate_bounded", lambda profile: (None, error))

    return _fail


@pytest.fixture
def databricks_login(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Stub the `databricks auth login` subprocess; it succeeds unless a test reconfigures it."""
    login = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(databricks_cli.subprocess, "run", login)
    return login


@pytest.fixture(autouse=True)
def human_cli_environment(monkeypatch):
    """Keep ordinary CLI assertions independent of the agent running the tests."""
    for name in (*AGENT_ENV_MARKERS, *AGENT_ENV_VALUES, "TERM_PROGRAM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AGENTBRICKS_DISABLE_AGENT_HINT", raising=False)
