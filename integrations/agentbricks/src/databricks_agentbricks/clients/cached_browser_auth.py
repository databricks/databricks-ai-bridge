"""Use the Python SDK's external-browser cache without ever starting OAuth consent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from databricks_agentkit._api_client import _AgentBricksApiClient

if TYPE_CHECKING:
    from databricks.sdk.config import Config
    from databricks.sdk.credentials_provider import CredentialsProvider


def _cached_credentials(config: Config) -> CredentialsProvider:
    from databricks.sdk import oauth

    # Group assumption was added after the minimum supported SDK (0.124).
    group_id = getattr(config, "group_id", None)
    # Match the SDK's external_browser cache identity, including Azure Entra profiles, scopes,
    # and group assumption. Using a different identity could reuse another profile's session.
    if config.client_id:
        client_id, client_secret = config.client_id, config.client_secret
        endpoints = config.databricks_oidc_endpoints
    elif config.azure_client_id:
        if group_id:
            raise ValueError("external-browser with Azure Entra ID cannot assume a group")
        client_id, client_secret = config.azure_client_id, config.azure_client_secret
        endpoints = oauth.get_azure_entra_id_workspace_endpoints(config.host)
    else:
        client_id, client_secret = "databricks-cli", None
        endpoints = config.databricks_oidc_endpoints
    if endpoints is None:
        raise ValueError("The workspace does not support OAuth")
    scopes = list(config.get_scopes())
    if not config.disable_oauth_refresh_token and "offline_access" not in scopes:
        scopes.append("offline_access")
    cache = oauth.TokenCache(
        host=config.host,
        oidc_endpoints=endpoints,
        client_id=client_id,
        client_secret=client_secret,
        redirect_url="http://localhost:8020",
        scopes=scopes,
        profile=config.profile,
        **({"group_id": group_id} if group_id else {}),
    )
    credentials = cache.load()
    if credentials is None:
        raise ValueError("No cached external-browser session is available")
    # A refresh failure propagates to the caller; unlike the SDK's interactive strategy there
    # is deliberately no fallback to OAuthClient, consent, or a browser listener here.
    credentials.token()
    return credentials(config)


def cached_browser_client(profile: Optional[str]) -> _AgentBricksApiClient:
    from databricks.sdk.credentials_provider import credentials_strategy

    strategy = credentials_strategy("external-browser", ["host"])(_cached_credentials)
    return _AgentBricksApiClient(profile, credentials_strategy=strategy)
