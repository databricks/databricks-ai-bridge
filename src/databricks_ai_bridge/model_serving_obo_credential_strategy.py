import logging
import os
import threading
from typing import Dict, Optional, Tuple

from databricks.sdk.core import Config
from databricks.sdk.credentials_provider import (
    CredentialsProvider,
    CredentialsStrategy,
    DefaultCredentials,
)

logger = logging.getLogger(__name__)


def _is_debug_mode():
    return os.getenv("OBO_DEBUG_MODE", "false") == "true"


def _log_debug_information(debug_str):
    if _is_debug_mode():
        logger.error(debug_str)


def is_gevent_running():
    """
    Check if gevent is running in async mode.

    Returns:
        bool: True if gevent is active and running, False otherwise
    """
    try:
        import gevent

        # Check if gevent monkey patching is active
        if hasattr(gevent, "socket") and hasattr(gevent.socket, "socket"):
            # Additional check to see if we're in a gevent context
            try:
                from gevent import getcurrent

                current = getcurrent()
                # If we get a gevent greenlet (not the main greenlet), gevent is running
                return current is not None and hasattr(current, "switch")
            except Exception:
                return False
        return False
    except ImportError:
        return False


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
    return is_in_model_serving_env == "true"


def _get_invokers_token_fallback():
    _log_debug_information("[Debug] Using Invokers Token Fallback")
    main_thread = threading.main_thread()
    thread_data = main_thread.__dict__
    invokers_token = None
    if "invokers_token" in thread_data:
        _log_debug_information("[Debug] Found Invokers Token in Thread Data")
        invokers_token = thread_data["invokers_token"]
    else:
        _log_debug_information("[Debug] Unable to find Invokers Token in Thread Data")
    return invokers_token


def _get_invokers_token_from_mlflowserving():
    try:
        from mlflowserving.scoring_server.agent_utils import fetch_obo_token

        _log_debug_information("[Debug] Retrieving OBO Token from Scoring Server")

        if _is_debug_mode():
            is_gevent = is_gevent_running()
            _log_debug_information(f"[Debug] Gevent Running: {is_gevent}")

        return fetch_obo_token()
    except ImportError:
        return _get_invokers_token_fallback()


def _get_invokers_token():
    invokers_token = _get_invokers_token_from_mlflowserving()
    if invokers_token is None:
        _log_debug_information("[Debug] Invokers token is None")
        raise RuntimeError(
            "Unable to read end user token in Databricks Model Serving. "
            "Please ensure you have specified UserAuthPolicy when logging the agent model "
            "and On Behalf of User Authorization for Agents is enabled in your workspace. "
            "If the issue persists, contact Databricks Support"
        )
    _log_debug_information("[Debug] Retrieved Invokers Token Successfully")
    return invokers_token


def get_databricks_host_token() -> Optional[Tuple[str, str]]:
    if not should_fetch_model_serving_environment_oauth():
        return None

    # read from DB_MODEL_SERVING_HOST_ENV_VAR if available otherwise MODEL_SERVING_HOST_ENV_VAR
    host = os.environ.get("DATABRICKS_MODEL_SERVING_HOST_URL") or os.environ.get(
        "DB_MODEL_SERVING_HOST_URL"
    )

    return (host, _get_invokers_token())


def model_serving_auth_visitor(cfg: Config) -> Optional[CredentialsProvider]:
    try:
        host, token = get_databricks_host_token()
        if token is None:
            raise ValueError(
                "Got malformed auth (empty token) when fetching auth implicitly available in Model Serving Environment. "
                "Please ensure you have specified UserAuthPolicy when logging the agent model and On Behalf of "
                "User Authorization for Agents is enabled in your workspace. If the issue persists, contact Databricks Support"
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
    This credential strategy is designed for authenticating the Databricks SDK in the model serving environment
    using user authorization (acting as the Databricks principal querying the serving endpoint).
    In the model serving environment, the strategy retrieves a downscoped user token or fails if no such token is available
    In any other environments, the class defaults to the DefaultCredentialStrategy.
    To use this credential strategy, instantiate the WorkspaceClient with the ModelServingUserCredentials strategy as follows:

    user_client = WorkspaceClient(credential_strategy = ModelServingUserCredentials())
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
            _log_debug_information("[Debug] Getting Invokers Credentials from Model Serving")
            header_factory = model_serving_auth_visitor(cfg)
            if not header_factory:
                raise ValueError(
                    "Unable to detect credentials for user authorization. "
                    "This error has two common causes: "
                    "(1) Improper OBO configuration - ensure you logged your model with a UserAuthPolicy and that the 'Agent Framework: On-Behalf-Of-User Authorization' preview is enabled in your workspace. "
                    "(2) WorkspaceClient instantiation outside of predict()/predict_stream() - ensure you instantiate the WorkspaceClient inside your predict() or predict_stream() function, not at model-loading time. "
                    "See https://docs.databricks.com/aws/en/generative-ai/agent-framework/authenticate-on-behalf-of-user for details. "
                    "If the issue persists, contact Databricks Support."
                )
            return header_factory
        else:
            return self.default_credentials(cfg)
