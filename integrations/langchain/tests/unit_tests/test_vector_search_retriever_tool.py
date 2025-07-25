import json
import os
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import mlflow
import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import ModelServingUserCredentials
from databricks.vector_search.utils import CredentialStrategy
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    DELTA_SYNC_INDEX_EMBEDDING_MODEL_ENDPOINT_NAME,
    INPUT_TEXTS,
    _get_index,
    mock_vs_client,
    mock_workspace_client,
)
from databricks_ai_bridge.vector_search_retriever_tool import FilterItem
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
)
from tests.utils.chat_models import llm, mock_client  # noqa: F401
from tests.utils.vector_search import (
    EMBEDDING_MODEL,
    embeddings,  # noqa: F401
)
from tests.utils.vector_search import (
    mock_client as mock_embeddings_client,  # noqa: F401
)


def init_vector_search_tool(
    index_name: str,
    columns: Optional[List[str]] = None,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    embedding: Optional[Embeddings] = None,
    text_column: Optional[str] = None,
    doc_uri: Optional[str] = None,
    primary_key: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> VectorSearchRetrieverTool:
    kwargs.update(
        {
            "index_name": index_name,
            "columns": columns,
            "tool_name": tool_name,
            "tool_description": tool_description,
            "embedding": embedding,
            "text_column": text_column,
            "doc_uri": doc_uri,
            "primary_key": primary_key,
            "filters": filters,
        }
    )
    if index_name != DELTA_SYNC_INDEX:
        kwargs.update(
            {
                "embedding": EMBEDDING_MODEL,
                "text_column": "text",
            }
        )
    return VectorSearchRetrieverTool(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_init(index_name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    assert isinstance(vector_search_tool, BaseTool)
    assert "'additionalProperties': true" not in str(vector_search_tool.args)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_chat_model_bind_tools(llm: ChatDatabricks, index_name: str) -> None:
    from langchain_core.messages import AIMessage

    vector_search_tool = init_vector_search_tool(index_name)
    llm_with_tools = llm.bind_tools([vector_search_tool])
    response = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
    assert isinstance(response, AIMessage)


def test_filters_are_passed_through() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)
    vector_search_tool._vector_store.similarity_search = MagicMock()

    vector_search_tool.invoke(
        {
            "query": "what cities are in Germany",
            "filters": [FilterItem(key="country", value="Germany")],
        }
    )
    vector_search_tool._vector_store.similarity_search.assert_called_once_with(
        query="what cities are in Germany",
        k=vector_search_tool.num_results,
        filter={"country": "Germany"},
        query_type=vector_search_tool.query_type,
    )


def test_filters_are_combined() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, filters={"city LIKE": "Berlin"})
    vector_search_tool._vector_store.similarity_search = MagicMock()

    vector_search_tool.invoke(
        {
            "query": "what cities are in Germany",
            "filters": [FilterItem(key="country", value="Germany")],
        }
    )
    vector_search_tool._vector_store.similarity_search.assert_called_once_with(
        query="what cities are in Germany",
        k=vector_search_tool.num_results,
        filter={"city LIKE": "Berlin", "country": "Germany"},
        query_type=vector_search_tool.query_type,
    )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize("columns", [None, ["id", "text"]])
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
@pytest.mark.parametrize("tool_description", [None, "Test tool for vector search"])
@pytest.mark.parametrize("embedding", [None, EMBEDDING_MODEL])
@pytest.mark.parametrize("text_column", [None, "text"])
def test_vector_search_retriever_tool_combinations(
    index_name: str,
    columns: Optional[List[str]],
    tool_name: Optional[str],
    tool_description: Optional[str],
    embedding: Optional[Any],
    text_column: Optional[str],
) -> None:
    if index_name == DELTA_SYNC_INDEX:
        embedding = None
        text_column = None

    vector_search_tool = init_vector_search_tool(
        index_name=index_name,
        columns=columns,
        tool_name=tool_name,
        tool_description=tool_description,
        embedding=embedding,
        text_column=text_column,
    )
    assert isinstance(vector_search_tool, BaseTool)
    result = vector_search_tool.invoke("Databricks Agent Framework")
    assert result is not None


def test_vector_search_retriever_tool_combinations() -> None:
    vector_search_tool = init_vector_search_tool(
        index_name=DELTA_SYNC_INDEX,
        doc_uri="uri",
        primary_key="id",
    )
    assert isinstance(vector_search_tool, BaseTool)
    result = vector_search_tool.invoke("Databricks Agent Framework")
    assert all(item.metadata.keys() == {"doc_uri", "chunk_id"} for item in result)
    assert all(item.page_content for item in result)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_vector_search_retriever_tool_description_generation(index_name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    assert vector_search_tool.name != ""
    assert vector_search_tool.description != ""
    assert vector_search_tool.name == index_name.replace(".", "__")
    assert (
        "A vector search-based retrieval tool for querying indexed embeddings."
        in vector_search_tool.description
    )
    assert vector_search_tool.args_schema.model_fields["query"] is not None
    assert vector_search_tool.args_schema.model_fields["query"].description == (
        "The string used to query the index with and identify the most similar "
        "vectors and return the associated documents."
    )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
def test_vs_tool_tracing(index_name: str, tool_name: Optional[str]) -> None:
    vector_search_tool = init_vector_search_tool(index_name, tool_name=tool_name)
    vector_search_tool._run("Databricks Agent Framework")
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    spans = trace.search_spans(name=tool_name or index_name, span_type=SpanType.RETRIEVER)
    assert len(spans) == 1
    inputs = json.loads(trace.to_dict()["data"]["spans"][0]["attributes"]["mlflow.spanInputs"])
    assert inputs["query"] == "Databricks Agent Framework"
    outputs = json.loads(trace.to_dict()["data"]["spans"][0]["attributes"]["mlflow.spanOutputs"])
    assert [d["page_content"] in INPUT_TEXTS for d in outputs]


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_vector_search_retriever_tool_resources(
    mock_embeddings_client,
    embeddings,
    index_name: str,
) -> None:
    text_column = "text"
    if index_name == DELTA_SYNC_INDEX:
        embeddings = None
        text_column = None

    vector_search_tool = VectorSearchRetrieverTool(
        index_name=index_name, embedding=embeddings, text_column=text_column
    )
    expected_resources = (
        [DatabricksVectorSearchIndex(index_name=index_name)]
        + ([DatabricksServingEndpoint(endpoint_name=embeddings.endpoint)] if embeddings else [])
        + (
            [
                DatabricksServingEndpoint(
                    endpoint_name=DELTA_SYNC_INDEX_EMBEDDING_MODEL_ENDPOINT_NAME
                )
            ]
            if index_name == DELTA_SYNC_INDEX
            else []
        )
    )
    assert [res.to_dict() for res in vector_search_tool.resources] == [
        res.to_dict() for res in expected_resources
    ]


@pytest.mark.parametrize("tool_name", [None, "valid_tool_name", "test_tool"])
def test_tool_name_validation_valid(tool_name: Optional[str]) -> None:
    index_name = "catalog.schema.index"
    tool = init_vector_search_tool(index_name, tool_name=tool_name)
    assert tool.tool_name == tool_name
    if tool_name:
        assert tool.name == tool_name


@pytest.mark.parametrize("tool_name", ["test.tool.name", "tool&name"])
def test_tool_name_validation_invalid(tool_name: str) -> None:
    index_name = "catalog.schema.index"
    with pytest.raises(ValueError):
        init_vector_search_tool(index_name, tool_name=tool_name)


@pytest.mark.parametrize(
    "index_name,name",
    [
        ("catalog.schema.index", "catalog__schema__index"),
        ("cata_log.schema_.index", "cata_log__schema___index"),
    ],
)
def test_index_name_to_tool_name(index_name: str, name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    assert vector_search_tool.name == name


def test_vector_search_client_model_serving_environment():
    with patch("os.path.isfile", return_value=True):
        # Simulate Model Serving Environment
        os.environ["IS_IN_DB_MODEL_SERVING_ENV"] = "true"

        # Fake credential token
        current_thread = threading.current_thread()
        thread_data = current_thread.__dict__
        thread_data["invokers_token"] = "abc"

        w = WorkspaceClient(
            host="testDogfod.com", credentials_strategy=ModelServingUserCredentials()
        )

        with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
            mock_instance = mockVSClient.return_value
            mock_instance.get_index.side_effect = _get_index
            with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
                vsTool = VectorSearchRetrieverTool(
                    index_name="test.delta_sync.index",
                    tool_description="desc",
                    workspace_client=w,
                )
                mockVSClient.assert_called_once_with(
                    disable_notice=True,
                    credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS,
                )


def test_vector_search_client_non_model_serving_environment():
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        mock_instance = mockVSClient.return_value
        mock_instance.get_index.side_effect = _get_index
        vsTool = VectorSearchRetrieverTool(
            index_name="test.delta_sync.index",
            tool_description="desc",
        )
        mockVSClient.assert_called_once_with(disable_notice=True)

    w = WorkspaceClient(host="testDogfod.com", token="fakeToken")
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
            mock_instance = mockVSClient.return_value
            mock_instance.get_index.side_effect = _get_index
            vsTool = VectorSearchRetrieverTool(
                index_name="test.delta_sync.index",
                tool_description="desc",
                workspace_client=w,
            )
            mockVSClient.assert_called_once_with(disable_notice=True)


def test_kwargs_are_passed_through() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, score_threshold=0.5)
    vector_search_tool._vector_store.similarity_search = MagicMock()

    vector_search_tool.invoke(
        {"query": "what cities are in Germany", "extra_param": "something random"},
    )
    vector_search_tool._vector_store.similarity_search.assert_called_once_with(
        query="what cities are in Germany",
        k=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        filter={},
        score_threshold=0.5,
        extra_param="something random",
    )
