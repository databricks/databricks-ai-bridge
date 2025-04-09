from unittest.mock import MagicMock

import pytest
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

from databricks_ai_bridge.utils.vector_search import IndexDetails
from databricks_ai_bridge.vector_search_retriever_tool import VectorSearchRetrieverToolMixin


class DummyVectorSearchRetrieverTool(VectorSearchRetrieverToolMixin):
    pass


index_name = "catalog.schema.index"


def make_mock_index_details(is_databricks_managed_embeddings=False, embedding_source_column=None):
    mock = MagicMock(spec=IndexDetails)
    mock.is_databricks_managed_embeddings = is_databricks_managed_embeddings
    mock.embedding_source_column = embedding_source_column or {}
    return mock


@pytest.mark.parametrize(
    "embedding_endpoint,index_details,resources",
    [
        (None, make_mock_index_details(False, {}), [DatabricksVectorSearchIndex(index_name)]),
        (
            "embedding_endpoint",
            make_mock_index_details(False, {}),
            [
                DatabricksVectorSearchIndex(index_name),
                DatabricksServingEndpoint("embedding_endpoint"),
            ],
        ),
        (
            None,
            make_mock_index_details(True, {"embedding_model_endpoint_name": "embedding_endpoint"}),
            [
                DatabricksVectorSearchIndex(index_name),
                DatabricksServingEndpoint("embedding_endpoint"),
            ],
        ),  # The following cases should not happen, but ensuring that they have reasonable behavior
        (
            "embedding_endpoint",
            make_mock_index_details(True, {"embedding_model_endpoint_name": "embedding_endpoint"}),
            [
                DatabricksVectorSearchIndex(index_name),
                DatabricksServingEndpoint("embedding_endpoint"),
            ],
        ),
        (
            "embedding_endpoint_1",
            make_mock_index_details(
                True, {"embedding_model_endpoint_name": "embedding_endpoint_2"}
            ),
            [
                DatabricksVectorSearchIndex(index_name),
                DatabricksServingEndpoint("embedding_endpoint_1"),
                DatabricksServingEndpoint("embedding_endpoint_2"),
            ],
        ),
        (None, make_mock_index_details(True, {}), [DatabricksVectorSearchIndex(index_name)]),
    ],
)
def test_get_resources(embedding_endpoint, index_details, resources):
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)
    assert tool._get_resources(index_name, embedding_endpoint, index_details) == resources
