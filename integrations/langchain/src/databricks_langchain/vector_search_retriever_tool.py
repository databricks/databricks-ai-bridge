from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator, PrivateAttr

from databricks_langchain import DatabricksVectorSearch
from databricks_langchain.utils import IndexDetails
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool


class VectorSearchRetrieverTool(BaseTool):
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with a Databricks Vector Search and provides a convenient interface
    for building a retriever tool for agents.
    """

    index_name: str = Field(..., description="The name of the index to use, format: 'catalog.schema.index'.")
    num_results: int = Field(10, description="The number of results to return.")
    columns: Optional[List[str]] = Field(None, description="Columns to return when doing the search.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply to the search.")
    query_type: str = Field("ANN", description="The type of query to run.")
    tool_name: Optional[str] = Field(None, description="The name of the retrieval tool.")
    tool_description: Optional[str] = Field(None, description="A description of the tool.")
    # TODO: Confirm if we can add these two to the API to support direct-access indexes or a delta-sync indexes with self-managed embeddings,
    text_column: Optional[str] = Field(None, description="If using a direct-access index or delta-sync index, specify the text column.")
    embedding: Optional[Embeddings] = Field(None, description="Embedding model for self-managed embeddings.")
    # TODO: Confirm if we can add this endpoint field
    endpoint: Optional[str] = Field(None, description="Endpoint for DatabricksVectorSearch.")

    # The BaseTool class requires 'name' and 'description' fields which we will populate in validate_tool_inputs()
    name: str = Field(default="", description="The name of the tool")
    description: str = Field(default="", description="The description of the tool")

    _vector_store: DatabricksVectorSearch = PrivateAttr()

    @model_validator(mode='after')
    def validate_tool_inputs(self):
        # Construct the vector store using provided params
        kwargs = {
            "index_name": self.index_name,
            "endpoint": self.endpoint,
            "embedding": self.embedding,
            "text_column": self.text_column,
            "columns": self.columns,
        }
        dbvs = DatabricksVectorSearch(**kwargs)
        self._vector_store = dbvs

        def get_tool_description():
            default_tool_description = "A vector search-based retrieval tool for querying indexed embeddings."
            index_details = IndexDetails(dbvs.index)
            if index_details.is_delta_sync_index():
                from databricks.sdk import WorkspaceClient

                source_table = index_details.index_spec.get('source_table', "")
                w = WorkspaceClient()
                source_table_comment = w.tables.get(full_name=source_table).comment
                if source_table_comment:
                    return (
                            default_tool_description +
                            f" The queried index uses the source table {source_table} with the description: " +
                            source_table_comment
                    )
                else:
                    return default_tool_description + f" The queried index uses the source table {source_table}"
            return default_tool_description

        self.name = self.tool_name or self.index_name
        self.description = self.tool_description or get_tool_description()

        return self


    def _run(self, query: str) -> str:
        return self._vector_store.similarity_search(
            query,
            k = self.num_results,
            filter = self.filters,
            query_type = self.query_type
        )
