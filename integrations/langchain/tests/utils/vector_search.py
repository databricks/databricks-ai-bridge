from typing import Any, Dict, Generator, List
from unittest import mock

import pytest
from databricks_ai_bridge.test_utils.vector_search import DEFAULT_VECTOR_DIMENSION
from langchain_core.embeddings import Embeddings

from databricks_langchain import DatabricksEmbeddings


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def __init__(self, dimension: int = DEFAULT_VECTOR_DIMENSION):
        super().__init__()
        self.dimension = dimension

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (self.dimension - 1) + [float(i)] for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (self.dimension - 1) + [float(0.0)]


EMBEDDING_MODEL = FakeEmbeddings()


def _mock_embeddings(endpoint: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": list(range(DEFAULT_VECTOR_DIMENSION)),
                "index": 0,
            }
            for _ in inputs["input"]
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


@pytest.fixture
def mock_client() -> Generator:
    client = mock.MagicMock()
    client.predict.side_effect = _mock_embeddings
    with mock.patch("mlflow.deployments.get_deploy_client", return_value=client):
        yield client


@pytest.fixture
def embeddings() -> DatabricksEmbeddings:
    return DatabricksEmbeddings(
        endpoint="text-embedding-3-small",
        documents_params={"fruit": "apple"},
        query_params={"fruit": "banana"},
    )
