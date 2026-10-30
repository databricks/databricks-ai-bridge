"""The Databricks workspace the system under test talks to (``--workspace live|fake``).

``live`` is a real workspace. ``fake`` is a local stand-in with no SDK access, so every requirement
that needs a real workspace is skipped rather than failed.
"""

from __future__ import annotations

import atexit
import enum
import os
import pathlib
import tempfile
from collections.abc import Mapping, Sequence
from typing import Protocol

import pytest

from harness.fake_workspace import FakeWorkspaceServer
from harness.run import RunConfig
from harness.workspace import Workspace


class WorkspaceKind(str, enum.Enum):
    LIVE = "live"
    FAKE = "fake"

    def __str__(self) -> str:
        return self.value


class TargetWorkspaceUnavailable(Exception):
    """The selected workspace cannot run in this environment; the suite skips instead of failing."""


class TargetWorkspace(Protocol):
    name: WorkspaceKind

    @property
    def base_env(self) -> Mapping[str, str]:
        """The ambient environment the ``agentbricks`` and ``databricks`` subprocesses start from."""
        ...

    @property
    def env(self) -> Mapping[str, str]:
        """Environment variables the subprocesses need, on top of ``base_env``."""
        ...

    @property
    def cli_args(self) -> Sequence[str]:
        """Global arguments placed after the CLI binary, e.g. ``--profile``."""
        ...

    @property
    def client(self) -> Workspace | None:
        """SDK access for cleanup and UC setup; None for the fake workspace."""
        ...


class LiveWorkspace:
    name = WorkspaceKind.LIVE

    def __init__(self, profile: str | None, run_config: RunConfig):
        self.profile = profile
        self._run_config = run_config
        self._client: Workspace | None = None

    @property
    def base_env(self) -> Mapping[str, str]:
        return os.environ

    @property
    def env(self) -> Mapping[str, str]:
        # The CLI looks up OAuth tokens by host, so several profiles on one host need the pin.
        return {"DATABRICKS_CONFIG_PROFILE": self.profile} if self.profile else {}

    @property
    def cli_args(self) -> Sequence[str]:
        return ["--profile", self.profile] if self.profile else []

    @property
    def client(self) -> Workspace:
        if self._client is None:
            self._client = Workspace(
                self.profile,
                warehouse_id=self._run_config.warehouse_id,
            )
        return self._client


_AMBIENT_PREFIXES = ("DATABRICKS_", "MLFLOW_", "AGENTBRICKS_")


class FakeWorkspace:
    """A local fake workspace: the CLI and the agent reach it through DATABRICKS_HOST/TOKEN only."""

    name = WorkspaceKind.FAKE
    cli_args: Sequence[str] = ()
    client: Workspace | None = None

    def __init__(self) -> None:
        self._server: FakeWorkspaceServer | None = None
        # An empty config file keeps the CLI from falling back to a real ~/.databrickscfg profile.
        self._config_dir = tempfile.TemporaryDirectory(prefix="fake-workspace-")
        self._config_file = pathlib.Path(self._config_dir.name) / "databrickscfg"
        self._config_file.write_text("")

    @property
    def server(self) -> FakeWorkspaceServer:
        # Started on first use: the leftover-cleanup hook builds a target it never talks to.
        if self._server is None:
            self._server = FakeWorkspaceServer()
            self._server.start()
            atexit.register(self._server.stop)
        return self._server

    @property
    def base_env(self) -> Mapping[str, str]:
        # Ambient credentials, tracking URIs or saved profiles would send the run to a real workspace.
        return {k: v for k, v in os.environ.items() if not k.startswith(_AMBIENT_PREFIXES)}

    @property
    def env(self) -> Mapping[str, str]:
        return {
            "DATABRICKS_HOST": self.server.url,
            "DATABRICKS_TOKEN": "fake-token",
            "DATABRICKS_AUTH_TYPE": "pat",
            "DATABRICKS_CONFIG_FILE": str(self._config_file),
            # No saved `agentbricks login` profile, which would make init seed a real profile.
            "AGENTBRICKS_CONFIG_HOME": self._config_dir.name,
        }


def require_workspace(target_workspace: TargetWorkspace, what: str) -> Workspace:
    """The live workspace's SDK client; skips the calling test or fixture on any other target."""
    workspace = target_workspace.client
    if workspace is None:
        pytest.skip(f"{what} needs a live workspace; got --workspace {target_workspace.name}")
    return workspace


def require_fake_workspace(target_workspace: TargetWorkspace, what: str) -> FakeWorkspaceServer:
    """The fake workspace's server; skips the calling test on any other target."""
    if not isinstance(target_workspace, FakeWorkspace):
        pytest.skip(f"{what} needs --workspace fake; got --workspace {target_workspace.name}")
    return target_workspace.server


def make_target_workspace(
    kind: WorkspaceKind, profile: str | None, run_config: RunConfig
) -> TargetWorkspace:
    if kind is WorkspaceKind.FAKE:
        return FakeWorkspace()
    return LiveWorkspace(profile, run_config)
