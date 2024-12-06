from typing import Any, Dict, Generator, List, Optional, Set

import pytest
from databricks.vector_search.client import VectorSearchIndex  # type: ignore

from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks
from tests.utils.vector_search import EMBEDDING_MODEL, DELTA_SYNC_INDEX, ALL_INDEX_NAMES, mock_vs_client
from tests.utils.chat_models import mock_client, llm
from langchain_core.tools import BaseTool

def init_vector_search_tool(
        index_name: str, columns: Optional[List[str]] = None
) -> VectorSearchRetrieverTool:
    kwargs: Dict[str, Any] = {
        "tool_name": "test_tool",
        "tool_description": "Test tool for vector search",
        "index_name": index_name,
        "columns": columns,
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