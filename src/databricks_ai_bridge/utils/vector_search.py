import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class IndexType(str, Enum):
    DIRECT_ACCESS = "DIRECT_ACCESS"
    DELTA_SYNC = "DELTA_SYNC"


class IndexDetails:
    """An utility class to store the configuration details of an index."""

    def __init__(self, index: Any):
        self._index_details = index.describe()

    @property
    def name(self) -> str:
        return self._index_details["name"]

    @property
    def schema(self) -> Optional[Dict]:
        if self.is_direct_access_index():
            schema_json = self.index_spec.get("schema_json")
            if schema_json is not None:
                return json.loads(schema_json)
        return None

    @property
    def primary_key(self) -> str:
        return self._index_details["primary_key"]

    @property
    def index_spec(self) -> Dict:
        return (
            self._index_details.get("delta_sync_index_spec", {})
            if self.is_delta_sync_index()
            else self._index_details.get("direct_access_index_spec", {})
        )

    @property
    def embedding_vector_column(self) -> Dict:
        if vector_columns := self.index_spec.get("embedding_vector_columns"):
            return vector_columns[0]
        return {}

    @property
    def embedding_source_column(self) -> Dict:
        if source_columns := self.index_spec.get("embedding_source_columns"):
            return source_columns[0]
        return {}

    def is_delta_sync_index(self) -> bool:
        return self._index_details["index_type"] == IndexType.DELTA_SYNC.value

    def is_direct_access_index(self) -> bool:
        return self._index_details["index_type"] == IndexType.DIRECT_ACCESS.value

    def is_databricks_managed_embeddings(self) -> bool:
        return self.is_delta_sync_index() and self.embedding_source_column.get("name") is not None


@dataclass
class RetrieverSchema:
    text_column: str = None
    doc_uri: Optional[str] = None
    chunk_id: Optional[str] = None
    other_columns: Optional[List[str]] = None


def get_metadata(columns: List[str], result: List[], doc_uri, chunk_id, other_columns, ignore_cols):
    metadata = {}
    for col, value in zip(columns[:-1], result[:-1]):
        if col == doc_uri:
            metadata["doc_uri"] = value
        elif col == chunk_id:
            metadata["chunk_id"] = value
        elif other_columns:
            if col in other_columns:
                metadata[col] = value
        elif col not in ignore_cols:
            metadata[col] = value
    return metadata


def parse_vector_search_response(
    search_resp: Dict,
    index_details: IndexDetails,
    retriever_schema: RetrieverSchema,
    ignore_cols: Optional[List[str]] = None,
    document_class: Any = dict,
) -> List[Tuple[Dict, float]]:
    """
    Parse the search response into a list of Documents with score.
    The document_class parameter is used to specify the class of the document to be created.
    """
    if ignore_cols is None:
        ignore_cols = []
    
    primary_key = index_details.primary_key
    text_column = retriever_schema.text_column
    doc_uri = retriever_schema.doc_uri
    chunk_id = retriever_schema.chunk_id
    other_columns = retriever_schema.other_columns

    ignore_cols.extend([primary_key, text_column])

    columns = [col["name"] for col in search_resp.get("manifest", dict()).get("columns", [])]
    docs_with_score = []
    
    for result in search_resp.get("result", dict()).get("data_array", []):
        page_content = result[columns.index(text_column)]
        
        metadata = get_metadata(columns, result, doc_uri, chunk_id, other_columns, ignore_cols)
        if doc_uri != primary_key:
            metadata[primary_key] = result[columns.index(primary_key)]
        
        score = result[-1]
        doc = document_class(page_content=page_content, metadata=metadata)
        docs_with_score.append((doc, score))
    
    return docs_with_score


def validate_and_get_text_column(text_column: Optional[str], index_details: IndexDetails) -> str:
    if index_details.is_databricks_managed_embeddings():
        index_source_column: str = index_details.embedding_source_column["name"]
        # check if input text column matches the source column of the index
        if text_column is not None:
            raise ValueError(
                f"The index '{index_details.name}' has the source column configured as "
                f"'{index_source_column}'. Do not pass the `text_column` parameter."
            )
        return index_source_column
    else:
        if text_column is None:
            raise ValueError("The `text_column` parameter is required for this index.")
        return text_column


def validate_and_get_return_columns(
    columns: List[str], text_column: str, index_details: IndexDetails
) -> List[str]:
    """
    Get a list of columns to retrieve from the index.
    If the index is direct-access index, validate the given columns against the schema.
    """
    # add primary key column and source column if not in columns
    if index_details.primary_key not in columns:
        columns.append(index_details.primary_key)
    if text_column and text_column not in columns:
        columns.append(text_column)

    # Validate specified columns are in the index
    if index_details.is_direct_access_index() and (index_schema := index_details.schema):
        if missing_columns := [c for c in columns if c not in index_schema]:
            raise ValueError(
                "Some columns specified in `columns` are not "
                f"in the index schema: {missing_columns}"
            )
    return columns
