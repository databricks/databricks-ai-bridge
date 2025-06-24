import threading

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
