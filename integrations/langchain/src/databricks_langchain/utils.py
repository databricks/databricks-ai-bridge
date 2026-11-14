from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI
from openai import AsyncOpenAI, OpenAI


def get_deployment_client(target_uri: str) -> Any:
    if (target_uri != "databricks") and (urlparse(target_uri).scheme != "databricks"):
        raise ValueError("Invalid target URI. The target URI must be a valid databricks URI.")

    try:
        from mlflow.deployments import get_deploy_client

        return get_deploy_client(target_uri)
    except ImportError as e:
        raise ImportError(
            "Failed to create the client. "
            "Please run `pip install mlflow` to install "
            "required dependencies."
        ) from e


def get_openai_client(workspace_client: Any = None, **kwargs) -> OpenAI:
    """Get an OpenAI client configured for Databricks.

    Args:
        workspace_client: Optional WorkspaceClient instance to use for authentication.
            If not provided, creates a default WorkspaceClient.
        **kwargs: Additional keyword arguments to pass to DatabricksOpenAI(),
            such as timeout, max_retries, use_ai_gateway, and use_ai_gateway_native_api.
    """
    from databricks.sdk import WorkspaceClient

    if workspace_client is None:
        workspace_client = WorkspaceClient()
    return DatabricksOpenAI(workspace_client=workspace_client, **kwargs)


def get_async_openai_client(workspace_client: Any = None, **kwargs) -> AsyncOpenAI:
    """Get an async OpenAI client configured for Databricks using databricks-openai.

    Args:
        workspace_client: Optional WorkspaceClient instance to use for authentication.
            If not provided, creates a default WorkspaceClient.
        **kwargs: Additional keyword arguments to pass to AsyncDatabricksOpenAI(),
            such as timeout and max_retries.
    """
    from databricks.sdk import WorkspaceClient

    # If workspace_client is provided, use it directly
    if workspace_client is not None:
        return AsyncDatabricksOpenAI(workspace_client=workspace_client, **kwargs)
    else:
        # Otherwise, create default workspace client and use it
        workspace_client = WorkspaceClient()
        return AsyncDatabricksOpenAI(workspace_client=workspace_client, **kwargs)


if TYPE_CHECKING:
    from databricks_langchain._vector_utils import Matrix as Matrix
    from databricks_langchain._vector_utils import cosine_similarity as cosine_similarity
    from databricks_langchain._vector_utils import (
        maximal_marginal_relevance as maximal_marginal_relevance,
    )


def __getattr__(name: str) -> object:
    # Preserve the existing utility imports without loading numerical libraries for chat.
    if name not in {"Matrix", "cosine_similarity", "maximal_marginal_relevance"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from databricks_langchain import _vector_utils

    value = getattr(_vector_utils, name)
    globals()[name] = value
    return value
