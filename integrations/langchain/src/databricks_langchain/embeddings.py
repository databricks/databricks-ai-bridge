from typing import Any, Dict, Iterator, List

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, PrivateAttr

from databricks_langchain.utils import get_deployment_client


class DatabricksEmbeddings(Embeddings, BaseModel):
    """Databricks embedding model integration.

    **Instantiate**:

        .. code-block:: python

            from databricks_langchain import DatabricksEmbeddings

            embed = DatabricksEmbeddings(
                endpoint="databricks-bge-large-en",
            )

    **Embed single text**:

        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

        .. code-block:: python

            [0.01605224609375, -0.0298309326171875, ...]

    """

    endpoint: str
    """Name of Databricks Model Serving endpoint to query."""
    target_uri: str = "databricks"
    """The target URI to use. Defaults to ``databricks``"""
    query_params: Dict[str, Any] = {}
    """The parameters to use for the query."""
    documents_params: Dict[str, Any] = {}
    """The parameters to use for documents."""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = get_deployment_client(self.target_uri)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, params=self.documents_params)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], params=self.query_params)[0]

    def _embed(self, texts: List[str], params: Dict[str, str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for txt in _chunk(texts, 20):
            resp = self._client.predict(
                endpoint=self.endpoint,
                inputs={"input": txt, **params},  # type: ignore[arg-type]
            )
            embeddings.extend(r["embedding"] for r in resp["data"])
        return embeddings


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]
