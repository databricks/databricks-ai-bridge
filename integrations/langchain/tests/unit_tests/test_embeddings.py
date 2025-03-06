"""Test Together AI embeddings."""

from databricks_ai_bridge.test_utils.vector_search import DEFAULT_VECTOR_DIMENSION
from mlflow.deployments import BaseDeploymentClient  # type: ignore[import-untyped]

from databricks_langchain import DatabricksEmbeddings
from tests.utils.vector_search import embeddings, mock_client  # noqa: F401


def test_embed_documents(
    mock_client: BaseDeploymentClient, embeddings: DatabricksEmbeddings
) -> None:
    documents = ["foo"] * 30
    output = embeddings.embed_documents(documents)
    assert len(output) == 30
    assert len(output[0]) == DEFAULT_VECTOR_DIMENSION
    assert mock_client.predict.call_count == 2
    assert all(
        call_arg[1]["inputs"]["fruit"] == "apple"
        for call_arg in mock_client().predict.call_args_list
    )


def test_embed_query(mock_client: BaseDeploymentClient, embeddings: DatabricksEmbeddings) -> None:
    query = "foo bar"
    output = embeddings.embed_query(query)
    assert len(output) == DEFAULT_VECTOR_DIMENSION
    mock_client.predict.assert_called_once()
    assert mock_client.predict.call_args[1] == {
        "endpoint": "text-embedding-3-small",
        "inputs": {"input": [query], "fruit": "banana"},
    }
