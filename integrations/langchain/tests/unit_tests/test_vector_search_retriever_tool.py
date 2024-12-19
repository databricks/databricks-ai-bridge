from typing import Any, Dict, Generator, List, Optional, Set

import pytest
from databricks.vector_search.client import VectorSearchIndex  # type: ignore

from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks
from tests.utils.vector_search import EMBEDDING_MODEL, DELTA_SYNC_INDEX, ALL_INDEX_NAMES, mock_vs_client, mock_workspace_client, mock_workspace_client
from tests.utils.chat_models import mock_client, llm
from langchain_core.tools import BaseTool
from langchain_core.embeddings import Embeddings

def init_vector_search_tool(
        index_name: str,
        columns: Optional[List[str]] = None,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        text_column: Optional[str] = None,
        endpoint: Optional[str] = None
) -> VectorSearchRetrieverTool:
    kwargs: Dict[str, Any] = {
        "index_name": index_name,
        "columns": columns,
        "tool_name": tool_name,
        "tool_description": tool_description,
        "embedding": embedding,
        "text_column": text_column,
        "endpoint": endpoint,
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
    assert isinstance(vector_search_tool, BaseTool)

@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_chat_model_bind_tools(llm: ChatDatabricks, index_name: str) -> None:
    from langchain_core.messages import AIMessage

    vector_search_tool = init_vector_search_tool(index_name)
    llm_with_tools = llm.bind_tools([vector_search_tool])
    response = llm_with_tools.invoke(
        "Which city is hotter today and which is bigger: LA or NY?"
    )
    assert isinstance(response, AIMessage)

@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize("columns", [None, ["id", "text"]])
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
@pytest.mark.parametrize("tool_description", [None, "Test tool for vector search"])
@pytest.mark.parametrize("embedding", [None, EMBEDDING_MODEL])
@pytest.mark.parametrize("text_column", [None, "text"])
@pytest.mark.parametrize("endpoint", [None, "test_endpoint"])
def test_vector_search_retriever_tool_combinations(
        index_name: str,
        columns: Optional[List[str]],
        tool_name: Optional[str],
        tool_description: Optional[str],
        embedding: Optional[Any],
        text_column: Optional[str],
        endpoint: Optional[str]
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
        endpoint=endpoint
    )
    assert isinstance(vector_search_tool, BaseTool)
    result = vector_search_tool.invoke("Databricks Agent Framework")
    assert result is not None