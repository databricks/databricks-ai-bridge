from typing import List
from langchain_core.embeddings import Embeddings
from databricks_ai_bridge.test_utils.vector_search import DEFAULT_VECTOR_DIMENSION

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
