from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from databricks.vector_search.client import VectorSearchIndex
from databricks_ai_bridge.test_utils.vector_search import (
    ALL_INDEX_NAMES,
    DEFAULT_VECTOR_DIMENSION,
    DELTA_SYNC_INDEX,
    EXAMPLE_SEARCH_RESPONSE,
    mock_vs_client,  # noqa: F401
    mock_workspace_client,  # noqa: F401
)
from databricks_ai_bridge.vector_search_retriever_tool import FilterItem
from google.adk.tools import FunctionTool

from databricks_google_adk import VectorSearchRetrieverTool


def fake_embedding_fn(text: str) -> list[float]:
    """Fake embedding function for testing."""
    return [1.0] * (DEFAULT_VECTOR_DIMENSION - 1) + [0.0]


def init_vector_search_tool(
    index_name: str,
    columns: Optional[List[str]] = None,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    embedding_fn=None,
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
            "embedding_fn": embedding_fn,
            "text_column": text_column,
            "filters": filters,
        }
    )
    if index_name != DELTA_SYNC_INDEX:
        kwargs.update(
            {
                "embedding_fn": fake_embedding_fn,
                "text_column": "text",
            }
        )
    return VectorSearchRetrieverTool(**kwargs)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_init(index_name: str) -> None:
    """Test that VectorSearchRetrieverTool initializes correctly."""
    vector_search_tool = init_vector_search_tool(index_name)
    assert isinstance(vector_search_tool, VectorSearchRetrieverTool)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_as_tool_returns_function_tool(index_name: str) -> None:
    """Test that as_tool() returns a FunctionTool."""
    vector_search_tool = init_vector_search_tool(index_name)
    adk_tool = vector_search_tool.as_tool()
    assert isinstance(adk_tool, FunctionTool)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_tool_name_generation(index_name: str) -> None:
    """Test that tool names are generated correctly."""
    vector_search_tool = init_vector_search_tool(index_name)
    expected_name = index_name.replace(".", "__")
    assert vector_search_tool._get_tool_name() == expected_name


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_custom_tool_name(index_name: str) -> None:
    """Test that custom tool names are respected."""
    custom_name = "my_custom_tool"
    vector_search_tool = init_vector_search_tool(index_name, tool_name=custom_name)
    assert vector_search_tool._get_tool_name() == custom_name


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_tool_description_generation(index_name: str) -> None:
    """Test that tool descriptions are generated correctly."""
    vector_search_tool = init_vector_search_tool(index_name)
    adk_tool = vector_search_tool.as_tool()
    # The function's docstring becomes the tool description
    assert adk_tool.func.__doc__ is not None
    assert "vector search" in adk_tool.func.__doc__.lower() or "search" in adk_tool.func.__doc__.lower()


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_custom_tool_description(index_name: str) -> None:
    """Test that custom tool descriptions are respected."""
    custom_description = "My custom tool description"
    vector_search_tool = init_vector_search_tool(index_name, tool_description=custom_description)
    adk_tool = vector_search_tool.as_tool()
    assert adk_tool.func.__doc__ == custom_description


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_search_execution(index_name: str) -> None:
    """Test that search can be executed."""
    vector_search_tool = init_vector_search_tool(index_name)
    results = vector_search_tool._search("test query")
    assert isinstance(results, list)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_as_tool_caching(index_name: str) -> None:
    """Test that as_tool() returns the same instance on repeated calls."""
    vector_search_tool = init_vector_search_tool(index_name)
    tool1 = vector_search_tool.as_tool()
    tool2 = vector_search_tool.as_tool()
    assert tool1 is tool2


def test_filters_are_passed_through() -> None:
    """Test that filters are correctly passed to the search."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)
    vector_search_tool._index = create_autospec(VectorSearchIndex, instance=True)
    vector_search_tool._index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE

    vector_search_tool._search(
        query="what cities are in Germany",
        filters=[FilterItem(key="country", value="Germany")],
    )
    vector_search_tool._index.similarity_search.assert_called_once()
    call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs
    assert call_kwargs["filters"] == {"country": "Germany"}


def test_filters_are_combined() -> None:
    """Test that runtime filters are combined with predefined filters."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, filters={"city LIKE": "Berlin"})
    vector_search_tool._index = create_autospec(VectorSearchIndex, instance=True)
    vector_search_tool._index.similarity_search.return_value = EXAMPLE_SEARCH_RESPONSE

    vector_search_tool._search(
        query="what cities are in Germany",
        filters=[FilterItem(key="country", value="Germany")],
    )
    call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs
    assert call_kwargs["filters"] == {"city LIKE": "Berlin", "country": "Germany"}


def test_dynamic_filter_creates_function_with_filters_param() -> None:
    """Test that dynamic_filter=True creates a tool with filters parameter."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, dynamic_filter=True)
    adk_tool = vector_search_tool.as_tool()

    # Check that the function accepts filters parameter
    import inspect
    sig = inspect.signature(adk_tool.func)
    assert "filters" in sig.parameters


def test_static_filter_creates_function_without_filters_param() -> None:
    """Test that dynamic_filter=False creates a tool without filters parameter."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, dynamic_filter=False)
    adk_tool = vector_search_tool.as_tool()

    # Check that the function does not accept filters parameter
    import inspect
    sig = inspect.signature(adk_tool.func)
    assert "filters" not in sig.parameters


def test_num_results_configuration() -> None:
    """Test that num_results is configurable."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, num_results=10)
    assert vector_search_tool.num_results == 10


def test_query_type_configuration() -> None:
    """Test that query_type is configurable."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, query_type="HYBRID")
    assert vector_search_tool.query_type == "HYBRID"


def test_embedding_fn_required_for_self_managed() -> None:
    """Test that embedding_fn is required for self-managed embeddings indexes."""
    # For non-delta-sync indexes, embedding_fn is required
    with pytest.raises(ValueError, match="embedding_fn is required"):
        tool = VectorSearchRetrieverTool(
            index_name="test.direct_access.index",
            text_column="text",
            # No embedding_fn provided
        )
        tool._search("test query")


def test_embedding_fn_not_allowed_for_databricks_managed() -> None:
    """Test that embedding_fn is not allowed for Databricks-managed embeddings."""
    tool = init_vector_search_tool(DELTA_SYNC_INDEX)
    tool.embedding_fn = fake_embedding_fn  # Try to set embedding_fn

    with pytest.raises(ValueError, match="Databricks-managed embeddings"):
        tool._search("test query")
