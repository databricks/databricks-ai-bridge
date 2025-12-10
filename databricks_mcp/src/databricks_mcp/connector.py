"""MCP OAuth Discovery and Dynamic Client Registration for Databricks.

This module provides functionality for discovering OAuth metadata and performing
Dynamic Client Registration (DCR) for MCP (Model Context Protocol) connections.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks_ai_bridge.utils.annotations import experimental

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 10
DEFAULT_PORT = "443"
WWW_AUTHENTICATE_HEADER = "WWW-Authenticate"
CONTENT_TYPE_JSON = "application/json"

# ---------------- Helpers ----------------


def _fetch_json(url: str, description: str) -> Dict[str, Any]:
    """Fetch and parse JSON from a URL.

    Args:
        url: The URL to fetch from
        description: Human-readable description for error messages

    Returns:
        Parsed JSON as a dictionary

    Raises:
        RuntimeError: If the request fails or JSON parsing fails
    """
    logger.debug(f"Fetching {description} from {url}")

    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Timeout while fetching {description} from {url}") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch {description} from {url}: {e}") from e

    try:
        data = resp.json()
        logger.debug(f"Successfully fetched {description}")
        return data
    except ValueError as e:
        raise RuntimeError(
            f"Failed to parse {description} from {url} as JSON: {e}. Response: {resp.text[:200]}"
        ) from e


def _parse_www_authenticate(header_value: str) -> Dict[str, str]:
    """Parse WWW-Authenticate header value into key-value parameters.

    Parses headers like:
        'Bearer resource_metadata="https://example.com/meta", scope="read write"'

    Args:
        header_value: The WWW-Authenticate header value

    Returns:
        Dictionary of parsed parameters from the header (quotes are removed from values)

    Example:
        >>> _parse_www_authenticate('Bearer scope="read write" realm="api"')
        {'scope': 'read write', 'realm': 'api'}
    """
    if not header_value:
        return {}

    # Split off the auth scheme (e.g., "Bearer") from parameters
    parts = header_value.split(" ", 1)
    params_part = parts[1] if len(parts) == 2 else parts[0]
    params = {}

    # Extract key="value" pairs using regex
    # Pattern: (\w+) captures parameter name, ([^"]*) captures value between quotes
    for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params_part):
        # Store without quotes: group(1) is key, group(2) is value
        params[m.group(1)] = m.group(2)

    logger.debug(f"Parsed WWW-Authenticate parameters: {list(params.keys())}")
    return params


def _try_resource_metadata_from_header(www_auth_header: str) -> Optional[str]:
    """Extract resource metadata URL from WWW-Authenticate header.

    Tries multiple parameter names in order of preference:
    1. resource_metadata (official spec)
    2. Common variations (authorization_uri, resource, as_uri, auth_uri)
    3. First URL found in the header (fallback)

    Args:
        www_auth_header: The WWW-Authenticate header value

    Returns:
        The resource metadata URL if found, None otherwise
    """
    params = _parse_www_authenticate(www_auth_header)

    # Official param per spec: resource_metadata
    if "resource_metadata" in params:
        logger.debug("Found resource_metadata in WWW-Authenticate header")
        return params["resource_metadata"]

    # Some servers use variations
    for key in ("authorization_uri", "resource", "as_uri", "auth_uri"):
        if key in params:
            logger.debug(f"Found {key} in WWW-Authenticate header")
            return params[key]

    # Very last resort: extract first URL found in the header
    # Pattern matches: http:// or https:// followed by any non-whitespace/comma/bracket chars
    m = re.search(r"https?://[^\s,>]+", www_auth_header)
    if m:
        logger.debug("Extracted URL from WWW-Authenticate header as fallback")
        return m.group(0)

    logger.debug("No resource metadata URL found in WWW-Authenticate header")
    return None


def _build_well_known_candidates(mcp_url: str) -> Tuple[str, str]:
    """Build candidate URLs for well-known OAuth protected resource metadata.

    Args:
        mcp_url: The MCP URL to derive well-known URLs from

    Returns:
        Tuple of two candidate URLs:
        - Path-specific well-known URL
        - Generic well-known URL
    """
    if not mcp_url:
        raise ValueError("mcp_url cannot be empty")

    parsed = urlparse(mcp_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid MCP URL: {mcp_url}")

    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.lstrip("/")  # e.g. "public/mcp"

    candidates = (
        f"{base}/.well-known/oauth-protected-resource/{path}",
        f"{base}/.well-known/oauth-protected-resource",
    )
    logger.debug(f"Built well-known candidates: {candidates}")
    return candidates


# ---------------- Protected Resource Discovery ----------------


def discover_protected_resource_metadata(mcp_url: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Discover OAuth Protected Resource metadata from an MCP URL.

    Attempts discovery in the following order:
    1. From WWW-Authenticate header in 401 response
    2. From path-specific well-known URL
    3. From generic well-known URL

    Args:
        mcp_url: The MCP endpoint URL

    Returns:
        Tuple of (Protected Resource metadata dictionary, WWW-Authenticate header value or None)

    Raises:
        ValueError: If mcp_url is invalid
        RuntimeError: If discovery fails via all methods
    """
    if not mcp_url:
        raise ValueError("mcp_url cannot be empty")

    logger.info(f"Discovering Protected Resource metadata for {mcp_url}")

    try:
        resp = requests.get(mcp_url, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to MCP URL {mcp_url}: {e}") from e

    if resp.status_code != 401:
        raise RuntimeError(
            f"Expected HTTP 401 from MCP URL for OAuth discovery, got {resp.status_code}. "
            f"The MCP endpoint may not be protected or properly configured. "
            f"Headers: {dict(resp.headers)}"
        )

    www_auth = resp.headers.get(WWW_AUTHENTICATE_HEADER)

    # 1. Header method
    if www_auth:
        url_from_header = _try_resource_metadata_from_header(www_auth)
        if url_from_header:
            try:
                metadata = _fetch_json(url_from_header, "Protected Resource metadata (from header)")
                return metadata, www_auth
            except RuntimeError as e:
                logger.warning(f"Failed to fetch from header URL: {e}")
                # fall through to well-known fallback

    # 2. Well-known fallback
    c1, c2 = _build_well_known_candidates(mcp_url)
    for candidate in (c1, c2):
        try:
            metadata = _fetch_json(candidate, "Protected Resource metadata (well-known)")
            return metadata, www_auth  # Return www_auth even if using well-known (may be None)
        except RuntimeError as e:
            logger.debug(f"Failed to fetch from {candidate}: {e}")
            continue

    raise RuntimeError(
        "Could not discover Protected Resource Metadata via header or well-known URIs. "
        "Ensure the MCP server supports OAuth discovery."
    )


# ---------------- AS Discovery ----------------


def discover_authorization_server_metadata(resource_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Discover Authorization Server metadata from Protected Resource metadata.

    Extracts the authorization server URL from the resource metadata and
    attempts to fetch the authorization server metadata from well-known endpoints.

    Args:
        resource_meta: Protected Resource metadata dictionary

    Returns:
        Authorization Server metadata as a dictionary

    Raises:
        RuntimeError: If authorization server URL cannot be determined or metadata cannot be fetched
    """
    if not resource_meta:
        raise ValueError("resource_meta cannot be empty")

    logger.info("Discovering Authorization Server metadata")

    # Extract authorization server base URL
    if "authorization_servers" in resource_meta and resource_meta["authorization_servers"]:
        as_base = resource_meta["authorization_servers"][0]
        logger.debug(f"Using authorization_servers[0]: {as_base}")
    elif "authorization_server" in resource_meta:
        as_base = resource_meta["authorization_server"]
        logger.debug(f"Using authorization_server: {as_base}")
    elif "issuer" in resource_meta:
        as_base = resource_meta["issuer"]
        logger.debug(f"Using issuer: {as_base}")
    else:
        raise RuntimeError(
            "Could not determine authorization server URL from Protected Resource metadata. "
            "Expected one of: authorization_servers, authorization_server, or issuer. "
            f"Available keys: {list(resource_meta.keys())}"
        )

    # Build candidate well-known URLs
    # Parse the as_base to extract base URL and path for multi-tenant support
    parsed = urlparse(as_base)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")

    candidates = []

    # Standard path appending (works for most single-tenant setups)
    candidates.append(urljoin(as_base.rstrip("/") + "/", ".well-known/oauth-authorization-server"))
    candidates.append(urljoin(as_base.rstrip("/") + "/", ".well-known/openid-configuration"))

    # Path insertion variants (for multi-tenant setups like https://auth.example.com/tenant1)
    if path:  # Only add these if there's a path component
        # OAuth 2.0 Authorization Server Metadata with path insertion
        candidates.append(f"{base_url}/.well-known/oauth-authorization-server{path}")
        # OpenID Connect Discovery 1.0 with path insertion
        candidates.append(f"{base_url}/.well-known/openid-configuration{path}")

    logger.debug(f"Trying authorization server metadata URLs: {candidates}")

    last_error = None
    for url in candidates:
        try:
            return _fetch_json(url, "Authorization Server metadata")
        except RuntimeError as e:
            logger.debug(f"Failed to fetch from {url}: {e}")
            last_error = e

    raise RuntimeError(
        f"Unable to fetch Authorization Server metadata from any candidate URL. "
        f"Last error: {last_error}"
    ) from last_error


# ---------------- DCR ----------------


def _select_oauth_scope(
    resource_meta: Dict[str, Any], www_auth_header: Optional[str] = None
) -> str:
    """Select OAuth scope following MCP specification priority order.

    Per MCP spec, clients SHOULD follow this priority order:
    1. Use 'scope' parameter from WWW-Authenticate header (if present)
    2. Use 'scopes_supported' from Protected Resource Metadata (if available)
    3. Omit scope parameter (return empty string if neither available)

    Args:
        resource_meta: Protected Resource metadata dictionary
        www_auth_header: Optional WWW-Authenticate header from 401 response

    Returns:
        Space-separated scope string, or empty string if no scope should be requested
    """
    # Priority 1: Check WWW-Authenticate header for scope parameter
    if www_auth_header:
        params = _parse_www_authenticate(www_auth_header)
        if "scope" in params:
            scope = params["scope"]
            logger.debug(f"Using scope from WWW-Authenticate header: {scope}")
            return scope

    # Priority 2: Use scopes_supported from Protected Resource Metadata
    if "scopes_supported" in resource_meta and resource_meta["scopes_supported"]:
        scopes = resource_meta["scopes_supported"]
        if isinstance(scopes, list):
            scope = " ".join(scopes)
        else:
            scope = str(scopes)
        logger.debug(f"Using scopes_supported from Protected Resource Metadata: {scope}")
        return scope

    # Priority 3: No scope parameter
    logger.debug("No scope information available; omitting scope parameter")
    return ""


def perform_dynamic_client_registration(
    as_meta: Dict[str, Any],
    resource_meta: Dict[str, Any],
    www_auth_header: Optional[str] = None,
    workspace_client: Optional[WorkspaceClient] = None,
) -> Dict[str, Any]:
    """Perform Dynamic Client Registration with the Authorization Server.

    Registers a new OAuth client with the authorization server using the
    registration endpoint from the AS metadata.

    Args:
        as_meta: Authorization Server metadata dictionary
        resource_meta: Protected Resource metadata dictionary (for scope selection)
        www_auth_header: Optional WWW-Authenticate header from 401 response (for scope selection)
        workspace_client: Optional WorkspaceClient instance. If None, a new one is created.

    Returns:
        Registration response with client credentials and endpoints

    Raises:
        ValueError: If as_meta is invalid
        RuntimeError: If DCR is not supported or registration fails
    """
    if not as_meta:
        raise ValueError("as_meta cannot be empty")

    logger.info("Performing Dynamic Client Registration")

    # Validate required endpoints
    reg_endpoint = as_meta.get("registration_endpoint")
    if not reg_endpoint:
        raise RuntimeError(
            "Authorization Server does NOT support Dynamic Client Registration "
            "(missing 'registration_endpoint'). "
            f"Available endpoints: {[k for k in as_meta.keys() if 'endpoint' in k]}"
        )

    authz = as_meta.get("authorization_endpoint")
    token = as_meta.get("token_endpoint")
    if not authz or not token:
        raise RuntimeError(
            "Authorization Server metadata missing required endpoints. "
            f"authorization_endpoint: {authz}, token_endpoint: {token}"
        )

    # Get Databricks workspace configuration
    w = workspace_client or WorkspaceClient()
    if not w.config.host:
        raise RuntimeError("WorkspaceClient host is not configured")

    redirect_uri = f"{w.config.host.rstrip('/')}/login/oauth/http.html"
    logger.debug(f"Using redirect URI: {redirect_uri}")

    # Select OAuth scope following MCP specification priority order
    selected_scope = _select_oauth_scope(resource_meta, www_auth_header=www_auth_header)

    # Build registration payload
    payload = {
        "client_name": "Databricks",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    logger.debug(f"Registration payload: {json.dumps(payload, indent=2)}")

    # Perform registration
    try:
        resp = requests.post(
            reg_endpoint,
            json=payload,
            headers={"Content-Type": CONTENT_TYPE_JSON},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Dynamic Client Registration request failed: {e}") from e

    # Parse registration response
    try:
        reg = resp.json()
    except ValueError as e:
        raise RuntimeError(
            f"Failed to parse DCR response JSON: {e}. Response: {resp.text[:200]}"
        ) from e

    # Validate response
    if "client_id" not in reg:
        raise RuntimeError(
            f"DCR response missing 'client_id'. Full response: {json.dumps(reg, indent=2)}"
        )

    logger.info(f"Successfully registered client: {reg.get('client_id')}")

    # Augment response with additional metadata
    reg["authorization_endpoint"] = authz
    reg["token_endpoint"] = token
    reg["redirect_uri"] = redirect_uri
    reg["registration_method"] = "dcr"
    reg["scope"] = selected_scope  # Store selected scope for UC connection

    return reg


def create_uc_connection(
    mcp_url: str,
    connection_name: str,
    dcr_result: Dict[str, Any],
    workspace_client: Optional[WorkspaceClient] = None,
) -> str:
    """Create a Unity Catalog HTTP connection for MCP.

    Args:
        mcp_url: The MCP endpoint URL
        connection_name: Name for the Unity Catalog connection (metastore-level resource)
        dcr_result: DCR result dictionary with client credentials and endpoints
        workspace_client: Optional WorkspaceClient instance. If None, a new one is created.

    Returns:
        The workspace URL to view the connection

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If connection creation fails
    """
    if not mcp_url:
        raise ValueError("mcp_url cannot be empty")
    if not connection_name:
        raise ValueError("connection_name cannot be empty")
    if not dcr_result:
        raise ValueError("dcr_result cannot be empty")

    # Parse and validate MCP URL early, before creating WorkspaceClient
    parsed = urlparse(mcp_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid MCP URL: {mcp_url}")

    logger.info(f"Creating Unity Catalog connection: {connection_name}")

    w = workspace_client or WorkspaceClient()

    host = f"{parsed.scheme}://{parsed.netloc}"
    base_path = parsed.path or "/"
    if not base_path.startswith("/"):
        base_path = "/" + base_path

    # Validate required fields from dcr_result
    client_id = dcr_result.get("client_id")
    if not client_id:
        raise ValueError("dcr_result missing required field: client_id")

    logger.debug(f"Connection config - host: {host}, base_path: {base_path}")

    try:
        w.connections.create(
            connection_type=catalog.ConnectionType.HTTP,
            name=connection_name,
            options={
                "host": host,
                "port": DEFAULT_PORT,
                "base_path": base_path,
                "oauth_credential_exchange_method": "header_and_body",
                "client_id": client_id,
                "client_secret": dcr_result.get("client_secret", ""),
                "authorization_endpoint": dcr_result.get("authorization_endpoint", ""),
                "token_endpoint": dcr_result.get("token_endpoint", ""),
                "oauth_scope": dcr_result.get("scope", ""),
                "is_mcp_connection": "true",
            },
        )
        logger.info(f"Successfully created Unity Catalog connection: {connection_name}")

        # Construct workspace URL for viewing the connection
        workspace_id = w.get_workspace_id()
        connection_url = f"{w.config.host}/explore/connections/{connection_name}?o={workspace_id}&activeTab=overview"
        return connection_url
    except Exception as e:
        raise RuntimeError(f"Failed to create Unity Catalog connection: {e}") from e


# ---------------- Public Entry Point ----------------


@experimental
def register_mcp_server_via_dcr(
    connection_name: str,
    mcp_url: str,
    workspace_client: Optional[WorkspaceClient] = None,
) -> str:
    """Register an MCP server via OAuth discovery and Dynamic Client Registration.

    This function performs the complete OAuth discovery and DCR flow:
    1. Check if Unity Catalog connection already exists (prevents duplicates)
    2. Discover Protected Resource Metadata (header + well-known fallback)
    3. Discover Authorization Server Metadata
    4. Select OAuth scope following MCP spec priority order:
       a. scope from WWW-Authenticate header (if present)
       b. scopes_supported from Protected Resource Metadata (if available)
       c. omit scope parameter (if neither available)
    5. Perform Dynamic Client Registration with Databricks redirect URI
    6. Create Unity Catalog connection with the registered client

    Args:
        connection_name: Name for the Unity Catalog connection
        mcp_url: The MCP endpoint URL
        workspace_client: Optional WorkspaceClient instance. If None, a new one is created.

    Returns:
        The workspace URL to view the connection

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If any step of the discovery or registration process fails

    Example:
        >>> connection_url = register_mcp_server_via_dcr(
        ...     connection_name="my_mcp_connection", mcp_url="https://mcp.example.com/api"
        ... )
        >>> print(connection_url)
        'https://workspace.cloud.databricks.com/explore/connections/my_mcp_connection?o=1234567890&activeTab=overview'
    """
    if not connection_name:
        raise ValueError("connection_name cannot be empty")
    if not mcp_url:
        raise ValueError("mcp_url cannot be empty")

    logger.info(f"Starting MCP client registration for connection: {connection_name}")
    logger.info(f"MCP URL: {mcp_url}")

    # Check if connection already exists to prevent duplicates
    w = workspace_client or WorkspaceClient()
    try:
        w.connections.get(connection_name)
        logger.info(f"Connection '{connection_name}' already exists, skipping DCR")
        # Construct workspace URL for the existing connection
        workspace_id = w.get_workspace_id()
        connection_url = f"{w.config.host}/explore/connections/{connection_name}?o={workspace_id}&activeTab=overview"
        return connection_url
    except Exception:
        # Connection doesn't exist, proceed with DCR
        logger.debug(f"Connection '{connection_name}' does not exist, proceeding with DCR")
        pass

    try:
        # Step 1: Discover Protected Resource Metadata
        prm, www_auth_header = discover_protected_resource_metadata(mcp_url)
        logger.debug(f"Protected Resource Metadata: {json.dumps(prm, indent=2)}")

        # Step 2: Discover Authorization Server Metadata
        as_meta = discover_authorization_server_metadata(prm)
        logger.debug(f"Authorization Server Metadata keys: {list(as_meta.keys())}")

        # Step 3: Perform Dynamic Client Registration
        dcr_result = perform_dynamic_client_registration(
            as_meta, prm, www_auth_header, workspace_client
        )

        # Step 4: Create Unity Catalog Connection
        connection_url = create_uc_connection(mcp_url, connection_name, dcr_result, w)

        logger.info("Successfully completed MCP client registration")
        return connection_url

    except Exception as e:
        logger.error(f"MCP client registration failed: {e}")
        raise
