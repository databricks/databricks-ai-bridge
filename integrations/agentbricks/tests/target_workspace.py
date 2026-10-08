"""The Databricks workspace the system under test talks to (``--workspace live|fake``).

``live`` is a real workspace. ``fake`` is a local stand-in with no SDK access, so every requirement
that needs a real workspace is skipped rather than failed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from common import RunConfig
from workspace_client import Workspace

WORKSPACE_KINDS = ("live", "fake")


class TargetWorkspaceUnavailable(Exception):
    """The selected workspace cannot run in this environment; the suite skips instead of failing."""


class TargetWorkspace(Protocol):
    name: str

    @property
    def env(self) -> Mapping[str, str]:
        """Environment variables the ``agentbricks`` and ``databricks`` subprocesses need."""
        ...

    @property
    def cli_args(self) -> Sequence[str]:
        """Global arguments placed after the CLI binary, e.g. ``--profile``."""
        ...

    @property
    def client(self) -> Workspace | None:
        """SDK access for grants, cleanup and UC setup; None for the fake workspace."""
        ...


class LiveWorkspace:
    name = "live"

    def __init__(self, profile: str | None, run_config: RunConfig):
        self.profile = profile
        self._run_config = run_config
        self._client: Workspace | None = None

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
                app_auth_profile=self._run_config.app_auth_profile,
                warehouse_id=self._run_config.warehouse_id,
                preprovisioned_app_catalog_access=self._run_config.preprovisioned_app_catalog_access,
            )
        return self._client


class FakeWorkspace:
    """Seam for the in-process fake: DATABRICKS_HOST/DATABRICKS_TOKEN env, no CLI args, no Workspace."""

    name = "fake"
    env: Mapping[str, str] = {}
    cli_args: Sequence[str] = ()
    client: Workspace | None = None

    def __init__(self, run_config: RunConfig):
        raise TargetWorkspaceUnavailable("fake workspace not implemented yet")


def make_target_workspace(name: str, profile: str | None, run_config: RunConfig) -> TargetWorkspace:
    if name == "fake":
        return FakeWorkspace(run_config)
    return LiveWorkspace(profile, run_config)
