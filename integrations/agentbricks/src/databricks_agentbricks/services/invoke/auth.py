"""Authentication adapter for endpoint invocation."""

from __future__ import annotations

from typing import Optional

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit._api_client import _workspace_client


class WorkspaceOAuthAuthenticator:
    """Resolve an OAuth header from one selected Databricks workspace profile."""

    def __init__(self, profile: Optional[str]) -> None:
        self._profile = profile

    def authorization_header(self) -> str:
        try:
            client = _workspace_client(self._profile)
            if client.config.auth_type == "pat":
                raise AgentCliError(
                    "Databricks Apps API routes require OAuth; the selected profile uses a PAT.",
                    hint="Authenticate the same workspace with `databricks auth login`.",
                )
            authorization = client.config.authenticate().get("Authorization")
        except AgentCliError:
            raise
        except Exception as exc:  # noqa: BLE001 - render auth failures without a traceback
            raise AgentCliError(f"Could not initialize endpoint authentication: {exc}.") from exc
        if not authorization:
            raise AgentCliError("Could not resolve an OAuth access token for the endpoint request.")
        return authorization
