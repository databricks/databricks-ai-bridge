import pytest
from unittest.mock import MagicMock

from databricks.sdk import WorkspaceClient

from databricks_ai_bridge.utils.auth import is_jwt, is_oauth_auth

# Dummy JWT token for testing purposes only (not a real secret)
_VALID_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"  # gitleaks:allow


class TestIsJwt:
    @pytest.mark.parametrize(
        "token,expected",
        [
            (_VALID_JWT, True),
            ("dapi1234567890abcdef", False),  # PAT token
            ("not.a.token", False),           # invalid base64 JSON parts
            ("", False),
        ],
    )
    def test_is_jwt(self, token, expected):
        assert is_jwt(token) == expected


class TestIsOauthAuth:
    @pytest.mark.parametrize(
        "oauth_side_effect,bearer_token,expected",
        [
            (None, None, True),                  # oauth_token succeeds
            (Exception("no oauth"), _VALID_JWT, True),   # JWT fallback → allowed
            (Exception("no oauth"), "dapi1234567890abcdef", False),  # PAT fallback → denied
        ],
    )
    def test_is_oauth_auth(self, oauth_side_effect, bearer_token, expected):
        mock_client = MagicMock(spec=WorkspaceClient)
        if oauth_side_effect:
            mock_client.config.oauth_token.side_effect = oauth_side_effect
            mock_client.client = MagicMock()
            mock_client.client.config.authenticate.return_value = {
                "Authorization": f"Bearer {bearer_token}"
            }
        assert is_oauth_auth(mock_client) is expected
