"""A small client over `databricks apps`, used by `deploy.py` and `endpoint.py`.

The `databricks apps get <name> -o json` call is the one read the CLI needs from the Apps
control plane, but three different pieces of information come out of it (service principal,
URL, compute state). Collapsing them behind a single `_get_json` accessor means the JSON is
fetched and parsed once per call site instead of being duplicated for each field. The
`databricks` runner is injected through the constructor (rather than imported as a module
global) so tests can substitute a fake runner instead of monkeypatching module-level
functions.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Callable, Optional

from databricks_agentbricks.databricks_cli import _databricks
from databricks_agentbricks.errors import AgentCliError

DatabricksRunner = Callable[..., subprocess.CompletedProcess]


class AppsClient:
    """Reads over `databricks apps`, behind a single `apps get -o json` accessor.

    The `runner` (the `databricks` CLI wrapper) is injected so callers - and tests - can
    supply a fake instead of monkeypatching a module-level function.
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
        """The app's compute state (e.g. RUNNING), or None if it can't be read."""
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
            hint=f"Check `agentbricks deployments get {name}`, then re-run deploy once it's running.",
        )
