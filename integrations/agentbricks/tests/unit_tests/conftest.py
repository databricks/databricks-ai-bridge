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

from databricks_agentbricks.cli import auth

_AMBIENT_ENV = ("DATABRICKS_CONFIG_PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN", "CI")


@pytest.fixture(autouse=True)
def databrickscfg_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """Path of the isolated (initially absent) `DATABRICKS_CONFIG_FILE`; the terminal is non-interactive."""
    for name in _AMBIENT_ENV:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path_factory.mktemp("databricks-config") / ".databrickscfg"
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(path))
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
    monkeypatch.setattr(auth, "_validate_bounded", validate)
    return validate
