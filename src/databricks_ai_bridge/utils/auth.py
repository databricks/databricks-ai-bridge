import jwt
from databricks.sdk import WorkspaceClient


def is_jwt(token: str) -> bool:
    try:
        jwt.decode(token, options={"verify_signature": False})
        return True
    except jwt.DecodeError:
        return False


def is_oauth_auth(workspace_client: WorkspaceClient) -> bool:
    """Check if the workspace client is using OAuth or JWT (M2M) authentication.

    First tries oauth_token(). If that fails, falls back to checking whether
    the bearer token is a JWT (e.g. M2M token), which is also considered valid.
    Returns False only for PAT or other non-OAuth auth types.
    """
    try:
        workspace_client.config.oauth_token()
        return True
    except Exception:
        headers = workspace_client.client.config.authenticate()
        token = headers["Authorization"].split("Bearer ")[1]
        return is_jwt(token)
