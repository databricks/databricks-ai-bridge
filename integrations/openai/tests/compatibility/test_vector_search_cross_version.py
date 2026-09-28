"""The v0.3-v0.5 OpenAI retriever contract against the current core package."""

from importlib.metadata import version
from unittest.mock import MagicMock, Mock, patch

import mlflow
import pytest
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    DELTA_SYNC_INDEX_EMBEDDING_MODEL_ENDPOINT_NAME,
    INPUT_TEXTS,
    mock_vs_client,
    mock_workspace_client,
)
from mlflow.entities import SpanType
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from packaging.version import Version
from pydantic import BaseModel

from databricks_openai import VectorSearchRetrieverTool


@pytest.fixture(autouse=True)
def mock_openai_client():
    # Only the external embedding service is mocked; the retriever and tracing are real.
    client = MagicMock()
    client.api_key = "fake_api_key"
    client.embeddings.create.return_value = Mock(data=[Mock(embedding=[0.1, 0.2, 0.3, 0.4])])
    with patch("openai.OpenAI", return_value=client):
        yield


@pytest.fixture(autouse=True)
def local_tracing(tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", previous_uri)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        experiment_id = mlflow.create_experiment(
            "openai-compatibility", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment_id)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        yield
    finally:
        mlflow.flush_trace_async_logging()
        mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize("index_name", sorted(ALL_INDEX_NAMES))
@pytest.mark.parametrize("columns", [None, ["id", "text"]])
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
@pytest.mark.parametrize("tool_description", [None, "Test tool for vector search"])
def test_vector_search_retriever_tool_init(
    index_name: str,
    columns: list[str] | None,
    tool_name: str | None,
    tool_description: str | None,
) -> None:
    managed_embeddings = index_name == DELTA_SYNC_INDEX
    embedding_model = None if managed_embeddings else "text-embedding-3-small"
    tool = VectorSearchRetrieverTool(
        index_name=index_name,
        columns=columns,
        tool_name=tool_name,
        tool_description=tool_description,
        text_column=None if managed_embeddings else "text",
        embedding_model_name=embedding_model,
    )
    assert isinstance(tool, BaseModel)

    endpoint_name = embedding_model
    # Of the historical refs in CI, only v0.5 (databricks-openai 0.4.0) declares the
    # managed embedding endpoint. Preserve each release's original resource assertion.
    if managed_embeddings and Version(version("databricks-openai")) >= Version("0.4.0"):
        endpoint_name = DELTA_SYNC_INDEX_EMBEDDING_MODEL_ENDPOINT_NAME
    expected_resources = [DatabricksVectorSearchIndex(index_name=index_name)] + (
        [DatabricksServingEndpoint(endpoint_name=endpoint_name)] if endpoint_name else []
    )
    assert tool.resources is not None
    assert [resource.to_dict() for resource in tool.resources] == [
        resource.to_dict() for resource in expected_resources
    ]

    docs = tool.execute(query="Databricks Agent Framework")
    assert docs is not None
    assert len(docs) == len(INPUT_TEXTS)
    assert sorted(document["page_content"] for document in docs) == sorted(INPUT_TEXTS)
    assert all("id" in document["metadata"] for document in docs)

    mlflow.flush_trace_async_logging()
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None
    trace = mlflow.get_trace(trace_id)
    assert trace is not None
    spans = trace.search_spans(name=tool_name or index_name, span_type=SpanType.RETRIEVER)
    assert len(spans) == 1
    span = spans[0]
    assert span.inputs["query"] == "Databricks Agent Framework"
    assert sorted(document["page_content"] for document in span.outputs) == sorted(INPUT_TEXTS)
