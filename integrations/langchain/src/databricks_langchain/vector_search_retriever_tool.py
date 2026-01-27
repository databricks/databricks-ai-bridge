import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union

from databricks_ai_bridge.utils.vector_search import IndexDetails

_logger = logging.getLogger(__name__)
from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    VectorSearchRetrieverToolInput,
    VectorSearchRetrieverToolMixin,
    vector_search_retriever_tool_trace,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from databricks_langchain import DatabricksEmbeddings
from databricks_langchain.multi_server_mcp_client import (
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from databricks_langchain.vectorstores import DatabricksVectorSearch


class VectorSearchRetrieverTool(BaseTool, VectorSearchRetrieverToolMixin):
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with Databricks Vector Search and provides a convenient interface
    for building a retriever tool for agents.

    **Note**: Any additional keyword arguments passed to the constructor will be passed along to
    `databricks.vector_search.client.VectorSearchIndex.similarity_search` when executing the tool. `See
    documentation <https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex.similarity_search>`_
    to see the full set of supported keyword arguments,
    e.g. `score_threshold`. Also, see documentation for
    :class:`~databricks_ai_bridge.vector_search_retriever_tool.VectorSearchRetrieverToolMixin` for additional supported constructor
    arguments not listed below, including `query_type` and `num_results`.

    WorkspaceClient instances with auth types PAT, OAuth-M2M (client ID and client secret), or model serving credential strategy will be used to instantiate the underlying VectorSearchClient.
    """

    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
        "Required for direct-access index or delta-sync index with "
        "self-managed embeddings.",
    )
    embedding: Optional[Embeddings] = Field(
        None, description="Embedding model for self-managed embeddings."
    )

    # The BaseTool class requires 'name' and 'description' fields which we will populate in validate_tool_inputs()
    name: str = Field(default="", description="The name of the tool")
    description: str = Field(default="", description="The description of the tool")
    args_schema: Type[BaseModel] = VectorSearchRetrieverToolInput

    _vector_store: DatabricksVectorSearch = PrivateAttr()
    _mcp_tool: Optional[LangChainBaseTool] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        kwargs = {
            "index_name": self.index_name,
            "embedding": self.embedding,
            "text_column": self.text_column,
            "doc_uri": self.doc_uri,
            "primary_key": self.primary_key,
            "columns": self.columns,
            "workspace_client": self.workspace_client,
            "include_score": self.include_score,
            "reranker": self.reranker,
        }
        dbvs = DatabricksVectorSearch(**kwargs)
        self._vector_store = dbvs

        self.name = self._get_tool_name()
        self.description = self.tool_description or self._get_default_tool_description(
            IndexDetails(dbvs.index)
        )
        self.resources = self._get_resources(
            self.index_name,
            (self.embedding.endpoint if isinstance(self.embedding, DatabricksEmbeddings) else None),
            IndexDetails(dbvs.index),
        )

        # Create args_schema based on dynamic_filter setting
        if self.dynamic_filter:
            self.args_schema = self._create_enhanced_input_model()
        else:
            self.args_schema = self._create_basic_input_model()

        return self

    def _create_or_get_mcp_tool(self) -> LangChainBaseTool:
        """Create or return existing MCP tool using LangChain MCP Server."""
        if self._mcp_tool is not None:
            return self._mcp_tool

        catalog, schema, index = self._parse_index_name()

        try:
            server = DatabricksMCPServer.from_vector_search(
                catalog=catalog,
                schema=schema,
                index_name=index,
                name=f"vs-{index}",
                workspace_client=self.workspace_client,
            )
            client = DatabricksMultiServerMCPClient([server])
        except Exception as e:
            self._handle_mcp_creation_error(e)

        tools = asyncio.run(client.get_tools())
        self._validate_mcp_tools(tools)

        self._mcp_tool = tools[0]
        return self._mcp_tool

    def _build_mcp_input(
        self,
        query: str,
        filters: Optional[Union[Dict[str, Any], List[FilterItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build input for MCP tool invocation."""
        mcp_input = self._build_mcp_params(filters, **kwargs)
        mcp_input["query"] = query
        return mcp_input

    def _parse_mcp_response(self, mcp_response: str) -> List[Document]:
        """Parse MCP tool response into LangChain Documents."""
        dicts = self._parse_mcp_response_to_dicts(mcp_response, strict=False)
        return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in dicts]

    def _execute_mcp_path(
        self,
        query: str,
        filters: Optional[Union[Dict[str, Any], List[FilterItem]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Execute vector search via LangChain MCP infrastructure."""
        try:
            mcp_tool = self._create_or_get_mcp_tool()
            mcp_input = self._build_mcp_input(query, filters, **kwargs)
            result = mcp_tool.invoke(mcp_input)
            return self._parse_mcp_response(result)
        except Exception as e:
            self._handle_mcp_execution_error(e)

    def _execute_direct_api_path(
        self,
        query: str,
        filters: Optional[Union[Dict[str, Any], List[FilterItem]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Execute vector search via direct DatabricksVectorSearch API."""
        kwargs = {**kwargs, **(self.model_extra or {})}
        # Normalize filters to dict format
        filters_dict = self._normalize_filters(filters)
        combined_filters = {**filters_dict, **(self.filters or {})}

        # Allow kwargs to override the default values upon invocation
        num_results = kwargs.pop("k", self.num_results)
        query_type = kwargs.pop("query_type", self.query_type)

        # Ensure that we don't have duplicate keys
        kwargs.update(
            {
                "query": query,
                "k": num_results,
                "filter": combined_filters,
                "query_type": query_type,
            }
        )
        return self._vector_store.similarity_search(**kwargs)

    @vector_search_retriever_tool_trace
    def _run(
        self,
        query: str,
        filters: Optional[Union[Dict[str, Any], List[FilterItem]]] = None,
        **kwargs,
    ) -> List[Document]:
        """Execute vector search with automatic routing."""
        index_details = IndexDetails(self._vector_store.index)

        if index_details.is_databricks_managed_embeddings():
            return self._execute_mcp_path(query, filters, **kwargs)
        else:
            return self._execute_direct_api_path(query, filters, **kwargs)
