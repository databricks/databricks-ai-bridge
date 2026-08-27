"""Public client for typed Mason memory and session resources."""

from typing import Optional

from databricks_mason._api_client import AgentApiClient
from databricks_mason.memory_store import MemoryStoreClient
from databricks_mason.session_store import SessionStoreClient


class DatabricksAgentClient:
    """Entry point for typed Databricks agent memory and session resources.

    Args:
        profile: Databricks config profile used when ``api_client`` is not supplied.
        api_client: Existing low-level client. This is primarily useful for tests and
            for callers that already share Mason authentication.
    """

    def __init__(
        self,
        profile: Optional[str] = None,
        *,
        api_client: Optional[AgentApiClient] = None,
    ) -> None:
        api = api_client if api_client is not None else AgentApiClient(profile=profile)
        self.memory_stores = MemoryStoreClient(api)
        self.session_stores = SessionStoreClient(api)
