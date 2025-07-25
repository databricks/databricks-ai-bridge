import os
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import ModelServingUserCredentials
from databricks.vector_search.client import VectorSearchIndex
from databricks.vector_search.utils import CredentialStrategy
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DEFAULT_VECTOR_DIMENSION,
    DELTA_SYNC_INDEX,
    mock_vs_client,
    mock_workspace_client,
)
from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    VectorSearchRetrieverToolInput,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from pydantic import Field

from databricks_llamaindex import VectorSearchRetrieverTool


class FakeEmbeddings(BaseEmbedding):
    """Fake embeddings functionality for testing."""

    dimension: int = Field(default=DEFAULT_VECTOR_DIMENSION)

    def get_text_embedding(self, text: str) -> List[float]:
        return [1.0] * (self.dimension - 1) + [0.0]

    def _aget_query_embedding(self):
        pass

    def _get_query_embedding(self):
        pass

    def _get_text_embedding(self):
        pass


EMBEDDING_MODEL = FakeEmbeddings()


def init_vector_search_tool(
    index_name: str,
    columns: Optional[List[str]] = None,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    embedding: Optional[BaseEmbedding] = None,
    text_column: Optional[str] = None,
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
    assert isinstance(vector_search_tool, FunctionTool)
    assert "'additionalProperties': true" not in str(
        vector_search_tool.metadata.fn_schema.model_json_schema()
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
    assert isinstance(vector_search_tool, FunctionTool)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_vector_search_retriever_tool_description_generation(index_name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    assert vector_search_tool.metadata.name != ""
    assert vector_search_tool.metadata.description != ""
    assert vector_search_tool.metadata.fn_schema == VectorSearchRetrieverToolInput
    assert vector_search_tool.metadata.name == index_name.replace(".", "__")
    assert (
        "A vector search-based retrieval tool for querying indexed embeddings."
        in vector_search_tool.metadata.description
    )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_vector_search_retriever_tool_bind_agent(index_name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    llm = OpenAI()
    assert ReActAgent.from_tools([vector_search_tool], llm=llm, verbose=True) is not None


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
            with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
                vsTool = VectorSearchRetrieverTool(
                    index_name="catalog.schema.my_index_name",
                    text_column="abc",
                    embedding_model_name="text-embedding-3-small",
                    tool_description="desc",
                    workspace_client=w,
                )
                mockVSClient.assert_called_once_with(
                    disable_notice=True,
                    credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS,
                )


def test_vector_search_client_non_model_serving_environment():
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        vsTool = VectorSearchRetrieverTool(
            index_name="catalog.schema.my_index_name",
            text_column="abc",
            embedding_model_name="text-embedding-3-small",
            tool_description="desc",
        )
        mockVSClient.assert_called_once_with(disable_notice=True, credential_strategy=None)

    w = WorkspaceClient(host="testDogfod.com", token="fakeToken")
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
            vsTool = VectorSearchRetrieverTool(
                index_name="catalog.schema.my_index_name",
                text_column="abc",
                embedding_model_name="text-embedding-3-small",
                tool_description="desc",
                workspace_client=w,
            )
            mockVSClient.assert_called_once_with(disable_notice=True, credential_strategy=None)


def test_kwargs_are_passed_through() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, score_threshold=0.5)
    vector_search_tool._index = create_autospec(VectorSearchIndex, instance=True)

    # extra_param is ignored because it isn't part of the signature for similarity_search
    vector_search_tool.call(
        query="what cities are in Germany", debug_level=2, extra_param="something random"
    )
    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text="what cities are in Germany",
        num_results=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        query_vector=None,
        filters={},
        score_threshold=0.5,
        debug_level=2,
    )


def test_filters_are_passed_through() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)
    vector_search_tool._index.similarity_search = MagicMock()

    vector_search_tool.call(
        query="what cities are in Germany", filters=[FilterItem(key="country", value="Germany")]
    )
    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text="what cities are in Germany",
        filters={"country": "Germany"},
        num_results=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        query_vector=None,
    )


def test_filters_are_combined() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, filters={"city LIKE": "Berlin"})
    vector_search_tool._index.similarity_search = MagicMock()

    vector_search_tool.call(
        query="what cities are in Germany", filters=[FilterItem(key="country", value="Germany")]
    )
    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text="what cities are in Germany",
        filters={"city LIKE": "Berlin", "country": "Germany"},
        num_results=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        query_vector=None,
    )
