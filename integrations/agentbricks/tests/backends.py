"""Where the system under test runs: what the CLI subprocess needs, and the workspace tests may touch.

``live`` is a real Databricks workspace. ``fake`` is an in-process stand-in that has no workspace, so
every requirement that needs one is skipped rather than failed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from common import RunConfig
from workspace_client import Workspace

BACKENDS = ("live", "fake")


class BackendUnavailable(Exception):
    """The selected backend cannot run in this environment; the suite skips instead of failing."""


class Backend(Protocol):
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
    def workspace(self) -> Workspace | None:
        """SDK access for grants, cleanup and UC setup; None when the backend has no workspace."""
        ...


class LiveBackend:
    name = "live"

    def __init__(self, profile: str | None, run_config: RunConfig):
        self.profile = profile
        self._run_config = run_config
        self._workspace: Workspace | None = None

    @property
    def env(self) -> Mapping[str, str]:
        # The CLI looks up OAuth tokens by host, so several profiles on one host need the pin.
        return {"DATABRICKS_CONFIG_PROFILE": self.profile} if self.profile else {}

    @property
    def cli_args(self) -> Sequence[str]:
        return ["--profile", self.profile] if self.profile else []

    @property
    def workspace(self) -> Workspace:
        if self._workspace is None:
            self._workspace = Workspace(
                self.profile,
                app_auth_profile=self._run_config.app_auth_profile,
                warehouse_id=self._run_config.warehouse_id,
                preprovisioned_app_catalog_access=self._run_config.preprovisioned_app_catalog_access,
            )
        return self._workspace


class FakeBackend:
    """Seam for the in-process fake: DATABRICKS_HOST/DATABRICKS_TOKEN env, no CLI args, no Workspace."""

    name = "fake"
    env: Mapping[str, str] = {}
    cli_args: Sequence[str] = ()
    workspace: Workspace | None = None

    def __init__(self, run_config: RunConfig):
        raise BackendUnavailable("fake backend not implemented yet")


def make_backend(name: str, profile: str | None, run_config: RunConfig) -> Backend:
    if name == "fake":
        return FakeBackend(run_config)
    return LiveBackend(profile, run_config)
