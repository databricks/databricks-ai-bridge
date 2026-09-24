"""Read/inspect helpers for `databricks apps` - one `apps get -o json` accessor.

Rather than the three duplicated `apps get -o json` call-sites that existed in
deploy.py, this module provides a single `_get_json` accessor that all inspection
methods share. The runner (the callable that shells out to the `databricks` CLI) is
injected via the constructor so tests pass a fake callable instead of monkeypatching
a module-global - which makes test failures local and self-describing.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Callable, Optional

from databricks_mason.databricks_cli import _databricks
from databricks_mason.errors import AgentCliError

DatabricksRunner = Callable[..., subprocess.CompletedProcess]


class AppsClient:
    """Read-only client over `databricks apps get`, with an injectable runner.

    All inspection methods share a single `_get_json` accessor that issues one
    `apps get -o json` call and caches nothing (each call is fresh). The runner
    defaults to the real `_databricks` CLI wrapper; tests inject a fake to stay
    hermetic without touching module globals.
    """

    def __init__(self, profile: Optional[str], *, runner: DatabricksRunner = _databricks) -> None:
        self._profile = profile
        self._run = runner

    def _get_json(self, name: str) -> Optional[dict]:
        result = self._run(
            ["apps", "get", name, "-o", "json"], self._profile, capture=True, check=False
        )
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

    def exists(self, name: str) -> bool:
        return (
            self._run(["apps", "get", name], self._profile, capture=True, check=False).returncode
            == 0
        )

    def service_principal(self, name: str) -> Optional[str]:
        """The app's service principal client id (its Postgres role identity), or None if unavailable."""
        data = self._get_json(name)
        return data.get("service_principal_client_id") if data else None

    def url(self, name: str) -> Optional[str]:
        """The deployed app's browsable URL, or None if it can't be read."""
        data = self._get_json(name)
        return (data.get("url") or None) if data else None

    def compute_state(self, name: str) -> Optional[str]:
        """The app's compute state (e.g. ACTIVE), or None if it can't be read."""
        data = self._get_json(name)
        return data.get("compute_status", {}).get("state") if data else None

    def wait_for_running(self, name: str, timeout_s: int = 300) -> None:
        """Block until a just-created app's compute is ACTIVE (or raise on timeout).

        `apps create` returns before compute is provisioned, but `apps deploy` requires the app to be
        ACTIVE - so a first deploy races without this wait.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.compute_state(name) == "ACTIVE":
                return
            time.sleep(5)
        raise AgentCliError(
            f"App '{name}' did not reach a running state within {timeout_s}s.",
            hint=f"Check `mason deployments get {name}`, then re-run deploy once it's running.",
        )
