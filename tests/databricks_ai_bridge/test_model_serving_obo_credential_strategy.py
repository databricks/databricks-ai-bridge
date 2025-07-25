import sys
import threading
from unittest.mock import MagicMock

from databricks.sdk.core import Config

from databricks_ai_bridge.model_serving_obo_credential_strategy import ModelServingUserCredentials


def test_agent_user_credentials(monkeypatch):
    # Guarantee that the tests defaults to env variables rather than config file.

    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "x")

    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DB_MODEL_SERVING_HOST_URL", "x")

    invokers_token_val = "databricks_invokers_token"
    current_thread = threading.current_thread()
    thread_data = current_thread.__dict__
    thread_data["invokers_token"] = invokers_token_val

    cfg = Config(credentials_strategy=ModelServingUserCredentials())
    assert cfg.auth_type == "model_serving_user_credentials"

    headers = cfg.authenticate()

    assert cfg.host == "x"
    assert headers.get("Authorization") == f"Bearer {invokers_token_val}"

    # Test updates of invokers token
    invokers_token_val = "databricks_invokers_token_v2"
    current_thread = threading.current_thread()
    thread_data = current_thread.__dict__
    thread_data["invokers_token"] = invokers_token_val

    headers = cfg.authenticate()
    assert cfg.host == "x"
    assert headers.get("Authorization") == f"Bearer {invokers_token_val}"

    # Test invokers token in child thread

    successful_authentication_event = threading.Event()

    def authenticate():
        try:
            cfg = Config(credentials_strategy=ModelServingUserCredentials())
            headers = cfg.authenticate()
            assert cfg.host == "x"
            assert headers.get("Authorization") == f"Bearer databricks_invokers_token_v2"
            successful_authentication_event.set()
        except Exception:
            successful_authentication_event.clear()

    thread = threading.Thread(target=authenticate)

    thread.start()
    thread.join()
    assert successful_authentication_event.is_set()


# If this credential strategy is being used in a non model serving environments then use default credential strategy instead
def test_agent_user_credentials_in_non_model_serving_environments(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "x")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")

    cfg = Config(credentials_strategy=ModelServingUserCredentials())
    assert (
        cfg.auth_type == "pat"
    )  # Auth type is PAT as it is no longer in a model serving environment

    headers = cfg.authenticate()

    assert cfg.host == "https://x"
    assert headers.get("Authorization") == "Bearer token"


def test_agent_user_credentials_with_mlflowserving_mock(monkeypatch):
    """Test authentication using mocked mlflowserving.scoring_server.agent_utils.fetch_obo_token"""

    # Guarantee that the tests defaults to env variables rather than config file.
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "x")
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DB_MODEL_SERVING_HOST_URL", "https://test-host.databricks.com")

    # Mock the mlflowserving module and fetch_obo_token function
    mock_mlflowserving = MagicMock()
    mock_agent_utils = MagicMock()
    mock_scoring_server = MagicMock()

    # Create the nested module structure
    mock_mlflowserving.scoring_server = mock_scoring_server
    mock_scoring_server.agent_utils = mock_agent_utils

    # Set up the mock function to return a token
    initial_token = "mlflow_obo_token_123"
    mock_agent_utils.fetch_obo_token.return_value = initial_token

    # Add the mock module to sys.modules
    monkeypatch.setitem(sys.modules, "mlflowserving", mock_mlflowserving)
    monkeypatch.setitem(sys.modules, "mlflowserving.scoring_server", mock_scoring_server)
    monkeypatch.setitem(sys.modules, "mlflowserving.scoring_server.agent_utils", mock_agent_utils)

    # Test authentication with the mocked mlflowserving
    cfg = Config(credentials_strategy=ModelServingUserCredentials())
    assert cfg.auth_type == "model_serving_user_credentials"

    headers = cfg.authenticate()
    assert cfg.host == "https://test-host.databricks.com"
    assert headers.get("Authorization") == f"Bearer {initial_token}"

    # Verify that fetch_obo_token was called
    mock_agent_utils.fetch_obo_token.assert_called()

    # Test token refresh - update the mock to return a new token
    updated_token = "mlflow_obo_token_456"
    mock_agent_utils.fetch_obo_token.return_value = updated_token

    # Authenticate again to test token refresh
    headers = cfg.authenticate()
    assert headers.get("Authorization") == f"Bearer {updated_token}"

    # Verify fetch_obo_token was called again
    assert mock_agent_utils.fetch_obo_token.call_count >= 2
