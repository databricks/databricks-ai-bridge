import pytest
from unittest.mock import MagicMock
from databricks_ai_bridge.utils.vector_search import IndexDetails, VectorSearchRetrieverToolMixin
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    INPUT_TEXTS,
    _get_index,
    mock_vs_client,
    mock_workspace_client,
)

class DummyRetriever(VectorSearchRetrieverToolMixin):
    pass

@pytest.fixture
def mock_index_details():
    mock = MagicMock(spec=IndexDetails)
    mock.is_databricks_managed_embeddings = False
    mock.embedding_source_column = {}
    return mock

def test_get_resources_index_only(mock_index_details):
    index_name = "catalog.schema.index"
    tool = DummyRetriever(index_name=index_name)
    resources = tool._get_resources(index_name, None, mock_index_details)

    assert resources == [DatabricksVectorSearchIndex(index_name)]

def test_get_resources_with_embedding_endpoint(mock_index_details):
    index_name = "catalog.schema.index"
    tool = DummyRetriever(index_name=index_name)
    resources = tool._get_resources(index_name, "embedding_endpoint", mock_index_details)

    assert resources == [
        DatabricksVectorSearchIndex(index_name),
        DatabricksServingEndpoint("embedding_endpoint")
    ]

def test_get_resources_with_managed_embeddings():
    index_name = "catalog.schema.index"
    mock = MagicMock(spec=IndexDetails)
    mock.is_databricks_managed_embeddings = True
    mock.embedding_source_column = {"embedding_model_endpoint_name": "embedding_endpoint"}

    tool = DummyRetriever(index_name=index_name)
    resources = tool._get_resources("catalog.schema.index", None, mock)

    assert resources == [
        DatabricksVectorSearchIndex(index_name),
        DatabricksServingEndpoint("embedding_endpoint")
    ]

def test_get_resources_with_duplicate_embedding_endpoints():
    index_name = "catalog.schema.index"
    mock = MagicMock(spec=IndexDetails)
    mock.is_databricks_managed_embeddings = True
    mock.embedding_source_column = {"embedding_model_endpoint_name": "embedding_endpoint"}

    tool = DummyRetriever(index_name=index_name)
    resources = tool._get_resources("catalog.schema.index", "embedding_endpoint", mock)

    assert resources == [
        DatabricksVectorSearchIndex(index_name),
        DatabricksServingEndpoint("embedding_endpoint")
    ]
