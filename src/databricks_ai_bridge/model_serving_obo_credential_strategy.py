import os
import threading
from typing import Dict, Optional, Tuple

from databricks.sdk.core import Config
from databricks.sdk.credentials_provider import (
    CredentialsProvider,
    CredentialsStrategy,
    DefaultCredentials,
)
import logging

logger = logging.getLogger(__name__)

_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH = "/var/credentials-secret/model-dependencies-oauth-token"

def should_fetch_model_serving_environment_oauth() -> bool:
        """
        Check whether this is the model serving environment
        Additionally check if the oauth token file path exists
        """

        is_in_model_serving_env = (
            os.environ.get("IS_IN_DB_MODEL_SERVING_ENV")
            or os.environ.get("IS_IN_DATABRICKS_MODEL_SERVING_ENV")
            or "false"
        )
        return is_in_model_serving_env == "true" and os.path.isfile(
            _MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH
        )

def _get_invokers_token():
    main_thread = threading.main_thread()
    thread_data = main_thread.__dict__
    invokers_token = None
    if "invokers_token" in thread_data:
        invokers_token = thread_data["invokers_token"]

    if invokers_token is None:
        raise RuntimeError("Unable to read Invokers Token in Databricks Model Serving")

    return invokers_token

def get_databricks_host_token() -> Optional[Tuple[str, str]]:
    if not should_fetch_model_serving_environment_oauth():
        return None

    # read from DB_MODEL_SERVING_HOST_ENV_VAR if available otherwise MODEL_SERVING_HOST_ENV_VAR
    host = os.environ.get("DATABRICKS_MODEL_SERVING_HOST_URL") or os.environ.get("DB_MODEL_SERVING_HOST_URL")

    return (host, _get_invokers_token())

def model_serving_auth_visitor(cfg: Config) -> Optional[CredentialsProvider]:
        try:
            host, token = get_databricks_host_token()
            if token is None:
                raise ValueError(
                    "Got malformed auth (empty token) when fetching auth implicitly available in Model Serving Environment. Please contact Databricks support"
                )
            if cfg.host is None:
                cfg.host = host
        except Exception as e:
            logger.warning(
                "Unable to get auth from Databricks Model Serving Environment",
                exc_info=e,
            )
            return None
        logger.info("Using Databricks Model Serving Authentication")

        def inner() -> Dict[str, str]:
            # Call here again to get the refreshed token
            _, token = get_databricks_host_token()
            return {"Authorization": f"Bearer {token}"}

        return inner


class ModelServingUserCredentials(CredentialsStrategy):
    """
    This credential strategy is designed for authenticating the Databricks SDK in the model serving environment using user-specific rights.
    In the model serving environment, the strategy retrieves a downscoped user token from the thread-local variable.
    In any other environments, the class defaults to the DefaultCredentialStrategy.
    To use this credential strategy, instantiate the WorkspaceClient with the ModelServingUserCredentials strategy as follows:

    invokers_client = WorkspaceClient(credential_strategy = ModelServingUserCredentials())
    """


    def __init__(self):
        self.default_credentials = DefaultCredentials()
    

    # Override
    def auth_type(self):
        if should_fetch_model_serving_environment_oauth():
            return "model_serving_user_credentials"
        else:
            return self.default_credentials.auth_type()
   

    # Override
    def __call__(self, cfg: Config) -> CredentialsProvider:
        if should_fetch_model_serving_environment_oauth():
            header_factory = model_serving_auth_visitor(cfg)
            if not header_factory:
                raise ValueError(
                    f"Unable to authenticate using model_serving_user_credentials in Databricks Model Serving Environment"
                )
            return header_factory
        else:
            return self.default_credentials(cfg)