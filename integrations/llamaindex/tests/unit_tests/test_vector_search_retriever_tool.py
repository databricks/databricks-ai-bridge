from typing import Any, Dict, List, Optional

import pytest
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DEFAULT_VECTOR_DIMENSION,
    DELTA_SYNC_INDEX,
    mock_vs_client,
    mock_workspace_client,
)
from databricks_ai_bridge.vector_search_retriever_tool import VectorSearchRetrieverToolInput
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
) -> VectorSearchRetrieverTool:
    kwargs: Dict[str, Any] = {
        "index_name": index_name,
        "columns": columns,
        "tool_name": tool_name,
        "tool_description": tool_description,
        "embedding": embedding,
        "text_column": text_column,
    }
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
    assert vector_search_tool.metadata.name == index_name
    assert (
        "A vector search-based retrieval tool for querying indexed embeddings."
        in vector_search_tool.metadata.description
    )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_vector_search_retriever_tool_bind_agent(index_name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    llm = OpenAI()
    assert ReActAgent.from_tools([vector_search_tool], llm=llm, verbose=True) is not None
