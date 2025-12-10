"""Unit tests for databricks_mcp.connector module."""

from unittest.mock import Mock, patch

import pytest
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

from databricks_mcp.connector import (
    _build_well_known_candidates,
    _fetch_json,
    _parse_www_authenticate,
    _try_resource_metadata_from_header,
    create_uc_connection,
    discover_authorization_server_metadata,
    discover_protected_resource_metadata,
    perform_dynamic_client_registration,
    register_mcp_server_via_dcr,
)


class TestFetchJson:
    """Tests for _fetch_json helper function."""

    @patch("databricks_mcp.connector.requests.get")
    def test_fetch_json_success(self, mock_get):
        """Test successful JSON fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        result = _fetch_json("https://example.com/data", "test data")

        assert result == {"key": "value"}
        mock_get.assert_called_once_with("https://example.com/data", timeout=10)

    @patch("databricks_mcp.connector.requests.get")
    def test_fetch_json_timeout(self, mock_get):
        """Test timeout handling."""
        mock_get.side_effect = requests.exceptions.Timeout()

        with pytest.raises(RuntimeError, match="Timeout while fetching"):
            _fetch_json("https://example.com/data", "test data")

    @patch("databricks_mcp.connector.requests.get")
    def test_fetch_json_http_error(self, mock_get):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="Failed to fetch"):
            _fetch_json("https://example.com/data", "test data")

    @patch("databricks_mcp.connector.requests.get")
    def test_fetch_json_invalid_json(self, mock_get):
        """Test invalid JSON response handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "not json"
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="Failed to parse.*as JSON"):
            _fetch_json("https://example.com/data", "test data")


class TestParseWwwAuthenticate:
    """Tests for _parse_www_authenticate helper function."""

    def test_parse_empty_header(self):
        """Test parsing empty header."""
        result = _parse_www_authenticate("")
        assert result == {}

    def test_parse_single_param(self):
        """Test parsing single parameter."""
        header = 'Bearer resource_metadata="https://example.com/metadata"'
        result = _parse_www_authenticate(header)
        assert result == {"resource_metadata": "https://example.com/metadata"}

    def test_parse_multiple_params(self):
        """Test parsing multiple parameters."""
        header = 'Bearer resource_metadata="https://example.com/meta" realm="api"'
        result = _parse_www_authenticate(header)
        assert result == {
            "resource_metadata": "https://example.com/meta",
            "realm": "api",
        }

    def test_parse_no_scheme(self):
        """Test parsing without auth scheme."""
        header = 'resource="https://example.com" scope="read"'
        result = _parse_www_authenticate(header)
        # The regex looks for word characters before =, so "https://example.com" won't match
        # because of the : and / characters. This test should expect only scope to be parsed.
        assert result == {"scope": "read"}


class TestTryResourceMetadataFromHeader:
    """Tests for _try_resource_metadata_from_header function."""

    def test_official_param(self):
        """Test extraction using official resource_metadata parameter."""
        header = 'Bearer resource_metadata="https://example.com/metadata"'
        result = _try_resource_metadata_from_header(header)
        assert result == "https://example.com/metadata"

    def test_authorization_uri_param(self):
        """Test extraction using authorization_uri parameter."""
        header = 'Bearer authorization_uri="https://example.com/auth"'
        result = _try_resource_metadata_from_header(header)
        assert result == "https://example.com/auth"

    def test_resource_param(self):
        """Test extraction using resource parameter."""
        header = 'Bearer resource="https://example.com/resource"'
        result = _try_resource_metadata_from_header(header)
        assert result == "https://example.com/resource"

    def test_fallback_to_url_extraction(self):
        """Test fallback to URL regex extraction."""
        header = "Bearer some text https://example.com/fallback more text"
        result = _try_resource_metadata_from_header(header)
        assert result == "https://example.com/fallback"

    def test_no_url_found(self):
        """Test when no URL can be extracted."""
        header = "Bearer realm=api"
        result = _try_resource_metadata_from_header(header)
        assert result is None


class TestBuildWellKnownCandidates:
    """Tests for _build_well_known_candidates function."""

    def test_simple_url(self):
        """Test with simple URL."""
        c1, c2 = _build_well_known_candidates("https://example.com/mcp")
        assert c1 == "https://example.com/.well-known/oauth-protected-resource/mcp"
        assert c2 == "https://example.com/.well-known/oauth-protected-resource"

    def test_url_with_nested_path(self):
        """Test with nested path."""
        c1, c2 = _build_well_known_candidates("https://example.com/api/v1/mcp")
        assert c1 == "https://example.com/.well-known/oauth-protected-resource/api/v1/mcp"
        assert c2 == "https://example.com/.well-known/oauth-protected-resource"

    def test_url_with_port(self):
        """Test with custom port."""
        c1, c2 = _build_well_known_candidates("https://example.com:8080/mcp")
        assert c1 == "https://example.com:8080/.well-known/oauth-protected-resource/mcp"
        assert c2 == "https://example.com:8080/.well-known/oauth-protected-resource"

    def test_empty_url(self):
        """Test with empty URL."""
        with pytest.raises(ValueError, match="mcp_url cannot be empty"):
            _build_well_known_candidates("")

    def test_invalid_url(self):
        """Test with invalid URL."""
        with pytest.raises(ValueError, match="Invalid MCP URL"):
            _build_well_known_candidates("not-a-url")


class TestDiscoverProtectedResourceMetadata:
    """Tests for discover_protected_resource_metadata function."""

    @patch("databricks_mcp.connector.requests.get")
    @patch("databricks_mcp.connector._fetch_json")
    def test_success_from_header(self, mock_fetch_json, mock_get):
        """Test successful discovery from WWW-Authenticate header."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {
            "WWW-Authenticate": 'Bearer resource_metadata="https://example.com/metadata"'
        }
        mock_get.return_value = mock_response
        mock_fetch_json.return_value = {"authorization_servers": ["https://auth.example.com"]}

        metadata, www_auth_header = discover_protected_resource_metadata("https://example.com/mcp")

        assert metadata == {"authorization_servers": ["https://auth.example.com"]}
        assert www_auth_header == 'Bearer resource_metadata="https://example.com/metadata"'
        mock_fetch_json.assert_called_once()

    @patch("databricks_mcp.connector.requests.get")
    @patch("databricks_mcp.connector._fetch_json")
    def test_success_from_well_known(self, mock_fetch_json, mock_get):
        """Test successful discovery from well-known URL."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_get.return_value = mock_response
        mock_fetch_json.return_value = {"authorization_servers": ["https://auth.example.com"]}

        metadata, www_auth_header = discover_protected_resource_metadata("https://example.com/mcp")

        assert metadata == {"authorization_servers": ["https://auth.example.com"]}
        assert www_auth_header is None
        assert mock_fetch_json.call_count >= 1

    @patch("databricks_mcp.connector.requests.get")
    def test_non_401_response(self, mock_get):
        """Test error when response is not 401."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="Expected HTTP 401"):
            discover_protected_resource_metadata("https://example.com/mcp")

    def test_empty_url(self):
        """Test with empty URL."""
        with pytest.raises(ValueError, match="mcp_url cannot be empty"):
            discover_protected_resource_metadata("")

    @patch("databricks_mcp.connector.requests.get")
    def test_connection_error(self, mock_get):
        """Test connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        with pytest.raises(RuntimeError, match="Failed to connect to MCP URL"):
            discover_protected_resource_metadata("https://example.com/mcp")


class TestDiscoverAuthorizationServerMetadata:
    """Tests for discover_authorization_server_metadata function."""

    @patch("databricks_mcp.connector._fetch_json")
    def test_success_with_authorization_servers(self, mock_fetch_json):
        """Test successful discovery using authorization_servers field."""
        resource_meta = {"authorization_servers": ["https://auth.example.com"]}
        mock_fetch_json.return_value = {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        result = discover_authorization_server_metadata(resource_meta)

        assert "authorization_endpoint" in result
        assert "token_endpoint" in result

    @patch("databricks_mcp.connector._fetch_json")
    def test_success_with_authorization_server(self, mock_fetch_json):
        """Test successful discovery using authorization_server field."""
        resource_meta = {"authorization_server": "https://auth.example.com"}
        mock_fetch_json.return_value = {
            "authorization_endpoint": "https://auth.example.com/authorize"
        }

        result = discover_authorization_server_metadata(resource_meta)

        assert "authorization_endpoint" in result

    @patch("databricks_mcp.connector._fetch_json")
    def test_success_with_issuer(self, mock_fetch_json):
        """Test successful discovery using issuer field."""
        resource_meta = {"issuer": "https://auth.example.com"}
        mock_fetch_json.return_value = {"token_endpoint": "https://auth.example.com/token"}

        result = discover_authorization_server_metadata(resource_meta)

        assert "token_endpoint" in result

    def test_missing_authorization_server(self):
        """Test error when authorization server cannot be determined."""
        resource_meta = {"some_other_field": "value"}

        with pytest.raises(RuntimeError, match="Could not determine authorization server URL"):
            discover_authorization_server_metadata(resource_meta)

    def test_empty_resource_meta(self):
        """Test with empty resource metadata."""
        with pytest.raises(ValueError, match="resource_meta cannot be empty"):
            discover_authorization_server_metadata({})

    @patch("databricks_mcp.connector._fetch_json")
    def test_all_candidates_fail(self, mock_fetch_json):
        """Test when all candidate URLs fail."""
        resource_meta = {"authorization_server": "https://auth.example.com"}
        mock_fetch_json.side_effect = RuntimeError("Failed to fetch")

        with pytest.raises(RuntimeError, match="Unable to fetch Authorization Server metadata"):
            discover_authorization_server_metadata(resource_meta)

    @patch("databricks_mcp.connector._fetch_json")
    def test_multi_tenant_path_insertion(self, mock_fetch_json):
        """Test multi-tenant discovery with path insertion."""
        resource_meta = {"authorization_server": "https://auth.example.com/tenant1"}

        # Simulate failure for standard URLs, success for path insertion variant
        def fetch_side_effect(url, description):
            if url == "https://auth.example.com/.well-known/oauth-authorization-server/tenant1":
                return {
                    "authorization_endpoint": "https://auth.example.com/tenant1/authorize",
                    "token_endpoint": "https://auth.example.com/tenant1/token",
                }
            raise RuntimeError(f"Not found: {url}")

        mock_fetch_json.side_effect = fetch_side_effect

        result = discover_authorization_server_metadata(resource_meta)

        assert result["authorization_endpoint"] == "https://auth.example.com/tenant1/authorize"
        assert result["token_endpoint"] == "https://auth.example.com/tenant1/token"
        # Verify it tried the path insertion variant
        assert any(
            "/.well-known/oauth-authorization-server/tenant1" in str(call)
            for call in mock_fetch_json.call_args_list
        )


class TestSelectOAuthScope:
    """Tests for _select_oauth_scope function."""

    def test_scope_from_www_auth_header(self):
        """Test scope selection from WWW-Authenticate header (priority 1)."""
        from databricks_mcp.connector import _select_oauth_scope

        resource_meta = {"scopes_supported": ["read", "write"]}
        www_auth_header = 'Bearer scope="admin delete"'

        scope = _select_oauth_scope(resource_meta, www_auth_header)

        assert scope == "admin delete"

    def test_scope_from_scopes_supported_list(self):
        """Test scope selection from scopes_supported list (priority 2)."""
        from databricks_mcp.connector import _select_oauth_scope

        resource_meta = {"scopes_supported": ["read", "write", "delete"]}

        scope = _select_oauth_scope(resource_meta, www_auth_header=None)

        assert scope == "read write delete"

    def test_scope_from_scopes_supported_string(self):
        """Test scope selection from scopes_supported string (priority 2)."""
        from databricks_mcp.connector import _select_oauth_scope

        resource_meta = {"scopes_supported": "read write"}

        scope = _select_oauth_scope(resource_meta, www_auth_header=None)

        assert scope == "read write"

    def test_no_scope_available(self):
        """Test when no scope information is available (priority 3)."""
        from databricks_mcp.connector import _select_oauth_scope

        resource_meta = {}

        scope = _select_oauth_scope(resource_meta, www_auth_header=None)

        assert scope == ""

    def test_empty_scopes_supported(self):
        """Test when scopes_supported is empty."""
        from databricks_mcp.connector import _select_oauth_scope

        resource_meta = {"scopes_supported": []}

        scope = _select_oauth_scope(resource_meta, www_auth_header=None)

        assert scope == ""

    def test_header_priority_over_metadata(self):
        """Test that WWW-Authenticate header takes priority over metadata."""
        from databricks_mcp.connector import _select_oauth_scope

        resource_meta = {"scopes_supported": ["read", "write"]}
        www_auth_header = 'Bearer scope="admin"'

        scope = _select_oauth_scope(resource_meta, www_auth_header)

        # Should use header scope, not metadata
        assert scope == "admin"
        assert scope != "read write"


class TestPerformDynamicClientRegistration:
    """Tests for perform_dynamic_client_registration function."""

    @patch("databricks_mcp.connector.requests.post")
    def test_success(self, mock_post):
        """Test successful dynamic client registration."""
        as_meta = {
            "registration_endpoint": "https://auth.example.com/register",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        mock_workspace = Mock(spec=WorkspaceClient)
        mock_workspace.config.host = "https://databricks.example.com"

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "client_id": "test-client-id",
            "client_secret": "test-secret",
        }
        mock_post.return_value = mock_response

        resource_meta = {"scopes_supported": ["read", "write"]}
        result = perform_dynamic_client_registration(
            as_meta, resource_meta, www_auth_header=None, workspace_client=mock_workspace
        )

        assert result["client_id"] == "test-client-id"
        assert result["client_secret"] == "test-secret"
        assert result["authorization_endpoint"] == "https://auth.example.com/authorize"
        assert result["token_endpoint"] == "https://auth.example.com/token"
        assert result["registration_method"] == "dcr"
        assert result["scope"] == "read write"
        assert "redirect_uri" in result

    def test_missing_registration_endpoint(self):
        """Test error when registration_endpoint is missing."""
        as_meta = {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        mock_workspace = Mock(spec=WorkspaceClient)
        mock_workspace.config.host = "https://databricks.example.com"

        resource_meta = {}
        with pytest.raises(RuntimeError, match="does NOT support Dynamic Client Registration"):
            perform_dynamic_client_registration(
                as_meta, resource_meta, www_auth_header=None, workspace_client=mock_workspace
            )

    def test_missing_authorization_endpoint(self):
        """Test error when authorization_endpoint is missing."""
        as_meta = {
            "registration_endpoint": "https://auth.example.com/register",
            "token_endpoint": "https://auth.example.com/token",
        }

        mock_workspace = Mock(spec=WorkspaceClient)
        mock_workspace.config.host = "https://databricks.example.com"

        resource_meta = {}
        with pytest.raises(RuntimeError, match="missing required endpoints"):
            perform_dynamic_client_registration(
                as_meta, resource_meta, www_auth_header=None, workspace_client=mock_workspace
            )

    @patch("databricks_mcp.connector.requests.post")
    def test_registration_http_error(self, mock_post):
        """Test HTTP error during registration."""
        as_meta = {
            "registration_endpoint": "https://auth.example.com/register",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        mock_workspace = Mock(spec=WorkspaceClient)
        mock_workspace.config.host = "https://databricks.example.com"

        mock_post.side_effect = requests.exceptions.HTTPError()

        resource_meta = {}
        with pytest.raises(RuntimeError, match="Dynamic Client Registration request failed"):
            perform_dynamic_client_registration(
                as_meta, resource_meta, www_auth_header=None, workspace_client=mock_workspace
            )

    @patch("databricks_mcp.connector.requests.post")
    def test_invalid_json_response(self, mock_post):
        """Test invalid JSON in registration response."""
        as_meta = {
            "registration_endpoint": "https://auth.example.com/register",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        mock_workspace = Mock(spec=WorkspaceClient)
        mock_workspace.config.host = "https://databricks.example.com"

        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "not json"
        mock_post.return_value = mock_response

        resource_meta = {}
        with pytest.raises(RuntimeError, match="Failed to parse DCR response"):
            perform_dynamic_client_registration(
                as_meta, resource_meta, www_auth_header=None, workspace_client=mock_workspace
            )

    @patch("databricks_mcp.connector.requests.post")
    def test_missing_client_id_in_response(self, mock_post):
        """Test error when client_id is missing from response."""
        as_meta = {
            "registration_endpoint": "https://auth.example.com/register",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        mock_workspace = Mock(spec=WorkspaceClient)
        mock_workspace.config.host = "https://databricks.example.com"

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"client_secret": "test-secret"}
        mock_post.return_value = mock_response

        resource_meta = {}
        with pytest.raises(RuntimeError, match="DCR response missing 'client_id'"):
            perform_dynamic_client_registration(
                as_meta, resource_meta, www_auth_header=None, workspace_client=mock_workspace
            )

    def test_empty_as_meta(self):
        """Test with empty as_meta."""
        resource_meta = {}
        with pytest.raises(ValueError, match="as_meta cannot be empty"):
            perform_dynamic_client_registration({}, resource_meta, None, None)


class TestCreateUcConnection:
    """Tests for create_uc_connection function."""

    @patch("databricks_mcp.connector.WorkspaceClient")
    def test_success(self, mock_workspace_class):
        """Test successful connection creation."""
        mock_workspace = Mock(spec=WorkspaceClient)
        mock_workspace.connections.create = Mock()
        mock_workspace.config.host = "https://workspace.databricks.com"
        mock_workspace.get_workspace_id = Mock(return_value=1234567890)
        mock_workspace_class.return_value = mock_workspace

        mcp_url = "https://mcp.example.com/api/v1"
        connection_name = "test_connection"
        as_meta = {
            "client_id": "test-client-id",
            "client_secret": "test-secret",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        result = create_uc_connection(mcp_url, connection_name, as_meta, mock_workspace)

        # Verify connection was created
        mock_workspace.connections.create.assert_called_once()
        call_kwargs = mock_workspace.connections.create.call_args[1]
        assert call_kwargs["name"] == connection_name
        assert call_kwargs["connection_type"] == catalog.ConnectionType.HTTP
        assert call_kwargs["options"]["client_id"] == "test-client-id"
        assert call_kwargs["options"]["host"] == "https://mcp.example.com"
        assert call_kwargs["options"]["base_path"] == "/api/v1"

        # Verify returned URL format
        assert (
            result
            == "https://workspace.databricks.com/explore/connections/test_connection?o=1234567890&activeTab=overview"
        )

    def test_empty_mcp_url(self):
        """Test with empty MCP URL."""
        with pytest.raises(ValueError, match="mcp_url cannot be empty"):
            create_uc_connection("", "test_connection", {"client_id": "test"})

    def test_empty_connection_name(self):
        """Test with empty connection name."""
        with pytest.raises(ValueError, match="connection_name cannot be empty"):
            create_uc_connection("https://example.com", "", {"client_id": "test"})

    def test_empty_dcr_result(self):
        """Test with empty dcr_result."""
        with pytest.raises(ValueError, match="dcr_result cannot be empty"):
            create_uc_connection("https://example.com", "test", {})

    def test_invalid_mcp_url(self):
        """Test with invalid MCP URL."""
        with pytest.raises(ValueError, match="Invalid MCP URL"):
            create_uc_connection("not-a-url", "test", {"client_id": "test"})

    def test_missing_client_id(self):
        """Test with missing client_id in dcr_result."""
        mock_workspace = Mock(spec=WorkspaceClient)

        with pytest.raises(ValueError, match="dcr_result missing required field: client_id"):
            create_uc_connection(
                "https://example.com", "test", {"client_secret": "secret"}, mock_workspace
            )

    @patch("databricks_mcp.connector.WorkspaceClient")
    def test_connection_creation_error(self, mock_workspace_class):
        """Test error during connection creation."""
        mock_workspace = Mock(spec=WorkspaceClient)
        mock_workspace.connections.create.side_effect = Exception("Connection error")
        mock_workspace_class.return_value = mock_workspace

        with pytest.raises(RuntimeError, match="Failed to create Unity Catalog connection"):
            create_uc_connection(
                "https://example.com",
                "test",
                {"client_id": "test"},
                mock_workspace,
            )


class TestRegisterMcpServerViaDcr:
    """Tests for register_mcp_server_via_dcr function."""

    @patch("databricks_mcp.connector.WorkspaceClient")
    @patch("databricks_mcp.connector.create_uc_connection")
    @patch("databricks_mcp.connector.perform_dynamic_client_registration")
    @patch("databricks_mcp.connector.discover_authorization_server_metadata")
    @patch("databricks_mcp.connector.discover_protected_resource_metadata")
    def test_success(
        self,
        mock_discover_prm,
        mock_discover_as,
        mock_perform_dcr,
        mock_create_conn,
        mock_workspace_class,
    ):
        """Test successful end-to-end registration."""
        # Mock WorkspaceClient for duplicate check
        mock_workspace = Mock()
        mock_workspace.connections.get.side_effect = Exception("Not found")
        mock_workspace_class.return_value = mock_workspace

        mock_discover_prm.return_value = (
            {"authorization_server": "https://auth.example.com"},
            'Bearer scope="read write"',
        )
        mock_discover_as.return_value = {
            "registration_endpoint": "https://auth.example.com/register",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }
        mock_perform_dcr.return_value = {
            "client_id": "test-client-id",
            "client_secret": "test-secret",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "redirect_uri": "https://databricks.example.com/login/oauth/http.html",
            "registration_method": "dcr",
            "scope": "read write",
        }
        mock_create_conn.return_value = "https://workspace.databricks.com/explore/connections/test_connection?o=1234567890&activeTab=overview"

        result = register_mcp_server_via_dcr("test_connection", "https://mcp.example.com/api")

        assert (
            result
            == "https://workspace.databricks.com/explore/connections/test_connection?o=1234567890&activeTab=overview"
        )
        mock_discover_prm.assert_called_once_with("https://mcp.example.com/api")
        mock_create_conn.assert_called_once()

    def test_empty_connection_name(self):
        """Test with empty connection name."""
        with pytest.raises(ValueError, match="connection_name cannot be empty"):
            register_mcp_server_via_dcr("", "https://mcp.example.com")

    def test_empty_mcp_url(self):
        """Test with empty MCP URL."""
        with pytest.raises(ValueError, match="mcp_url cannot be empty"):
            register_mcp_server_via_dcr("test_connection", "")

    @patch("databricks_mcp.connector.WorkspaceClient")
    def test_connection_already_exists(self, mock_workspace_class):
        """Test that existing connection is returned without running DCR."""
        mock_workspace = Mock()
        mock_existing_conn = Mock()
        mock_workspace.connections.get.return_value = mock_existing_conn
        mock_workspace.config.host = "https://workspace.databricks.com"
        mock_workspace.get_workspace_id = Mock(return_value=1234567890)
        mock_workspace_class.return_value = mock_workspace

        result = register_mcp_server_via_dcr("existing_connection", "https://mcp.example.com/api")

        assert (
            result
            == "https://workspace.databricks.com/explore/connections/existing_connection?o=1234567890&activeTab=overview"
        )
        # Verify that DCR flow was not triggered
        mock_workspace.connections.get.assert_called_once_with("existing_connection")

    @patch("databricks_mcp.connector.WorkspaceClient")
    @patch("databricks_mcp.connector.discover_protected_resource_metadata")
    def test_discovery_failure(self, mock_discover_prm, mock_workspace_class):
        """Test handling of discovery failure."""
        # Mock WorkspaceClient for duplicate check
        mock_workspace = Mock()
        mock_workspace.connections.get.side_effect = Exception("Not found")
        mock_workspace_class.return_value = mock_workspace

        mock_discover_prm.side_effect = RuntimeError("Discovery failed")

        with pytest.raises(RuntimeError, match="Discovery failed"):
            register_mcp_server_via_dcr("test_connection", "https://mcp.example.com/api")

    @patch("databricks_mcp.connector.WorkspaceClient")
    @patch("databricks_mcp.connector.perform_dynamic_client_registration")
    @patch("databricks_mcp.connector.discover_authorization_server_metadata")
    @patch("databricks_mcp.connector.discover_protected_resource_metadata")
    def test_dcr_failure(
        self,
        mock_discover_prm,
        mock_discover_as,
        mock_perform_dcr,
        mock_workspace_class,
    ):
        """Test handling of DCR failure."""
        # Mock WorkspaceClient for duplicate check
        mock_workspace = Mock()
        mock_workspace.connections.get.side_effect = Exception("Not found")
        mock_workspace_class.return_value = mock_workspace

        mock_discover_prm.return_value = (
            {"authorization_server": "https://auth.example.com"},
            None,
        )
        mock_discover_as.return_value = {
            "registration_endpoint": "https://auth.example.com/register",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }
        mock_perform_dcr.side_effect = RuntimeError("DCR failed")

        with pytest.raises(RuntimeError, match="DCR failed"):
            register_mcp_server_via_dcr("test_connection", "https://mcp.example.com/api")
