import uuid
from typing import Generator, List, Optional
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from databricks.vector_search.client import VectorSearchIndex  # type: ignore
from langchain_core.embeddings import Embeddings

INPUT_TEXTS = ["foo", "bar", "baz"]
DEFAULT_VECTOR_DIMENSION = 4


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


### Dummy similarity_search() Response ###
EXAMPLE_SEARCH_RESPONSE = {
    "manifest": {
        "column_count": 3,
        "columns": [
            {"name": "id"},
            {"name": "text"},
            {"name": "text_vector"},
            {"name": "score"},
        ],
    },
    "result": {
        "row_count": len(INPUT_TEXTS),
        "data_array": sorted(
            [
                [str(uuid.uuid4()), s, e, 0.5]
                for s, e in zip(INPUT_TEXTS, EMBEDDING_MODEL.embed_documents(INPUT_TEXTS))
            ],
            key=lambda x: x[2],  # type: ignore
            reverse=True,
        ),
    },
    "next_page_token": "",
}


### Dummy Indices ####

ENDPOINT_NAME = "test-endpoint"
DIRECT_ACCESS_INDEX = "test.direct_access.index"
DELTA_SYNC_INDEX = "test.delta_sync.index"
DELTA_SYNC_SELF_MANAGED_EMBEDDINGS_INDEX = "test.delta_sync_self_managed.index"
ALL_INDEX_NAMES = {
    DIRECT_ACCESS_INDEX,
    DELTA_SYNC_INDEX,
    DELTA_SYNC_SELF_MANAGED_EMBEDDINGS_INDEX,
}

INDEX_DETAILS = {
    DELTA_SYNC_INDEX: {
        "name": DELTA_SYNC_INDEX,
        "endpoint_name": ENDPOINT_NAME,
        "index_type": "DELTA_SYNC",
        "primary_key": "id",
        "delta_sync_index_spec": {
            "source_table": "ml.llm.source_table",
            "pipeline_type": "CONTINUOUS",
            "embedding_source_columns": [
                {
                    "name": "text",
                    "embedding_model_endpoint_name": "openai-text-embedding",
                }
            ],
        },
    },
    DELTA_SYNC_SELF_MANAGED_EMBEDDINGS_INDEX: {
        "name": DELTA_SYNC_SELF_MANAGED_EMBEDDINGS_INDEX,
        "endpoint_name": ENDPOINT_NAME,
        "index_type": "DELTA_SYNC",
        "primary_key": "id",
        "delta_sync_index_spec": {
            "source_table": "ml.llm.source_table",
            "pipeline_type": "CONTINUOUS",
            "embedding_vector_columns": [
                {
                    "name": "text_vector",
                    "embedding_dimension": DEFAULT_VECTOR_DIMENSION,
                }
            ],
        },
    },
    DIRECT_ACCESS_INDEX: {
        "name": DIRECT_ACCESS_INDEX,
        "endpoint_name": ENDPOINT_NAME,
        "index_type": "DIRECT_ACCESS",
        "primary_key": "id",
        "direct_access_index_spec": {
            "embedding_vector_columns": [
                {
                    "name": "text_vector",
                    "embedding_dimension": DEFAULT_VECTOR_DIMENSION,
                }
            ],
            "schema_json": f"{{"
            f'"{"id"}": "int", '
            f'"feat1": "str", '
            f'"feat2": "float", '
            f'"text": "string", '
            f'"{"text_vector"}": "array<float>"'
            f"}}",
        },
    },
}


@pytest.fixture(autouse=True)
def mock_vs_client() -> Generator:
    def _get_index(
        endpoint_name: Optional[str] = None,
        index_name: str = None,  # type: ignore
    ) -> MagicMock:
        index = MagicMock(spec=VectorSearchIndex)
        index.describe.return_value = INDEX_DETAILS[index_name]
        index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE
        return index

    mock_client = MagicMock()
    mock_client.get_index.side_effect = _get_index
    with mock.patch(
        "databricks.vector_search.client.VectorSearchClient",
        return_value=mock_client,
    ):
        yield


@pytest.fixture(autouse=True)
def mock_workspace_client() -> Generator:
    def _get_table_comment(full_name: str) -> MagicMock:
        table = MagicMock()
        table.comment = "Mocked table comment"
        return table

    mock_client = MagicMock()
    mock_client.tables.get.side_effect = _get_table_comment
    with patch(
        "databricks.sdk.WorkspaceClient",
        return_value=mock_client,
    ):
        yield
