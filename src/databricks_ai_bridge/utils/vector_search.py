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
    primary_key: Optional[str] = None
    other_columns: Optional[List[str]] = None


def get_metadata(
    columns: List[str],
    result: List[Any],
    retriever_schema: RetrieverSchema,
    ignore_cols: List[str],
    include_score: bool,
):
    """
    This function constructs a metadata dictionary by mapping column names to their corresponding values
    from the result row, with special handling for the provided retriever schema.

    Args:
        columns (List[str]): The list of column names in the result, including a final score column.
        result (List[Any]): The row of data values corresponding to the columns.
        retriever_schema: An object that defines which columns represent `doc_uri`, `primary_key`,
                          and optionally other columns to include.
        ignore_cols (List[str]): List of column names to exclude from the metadata. Usually contains
                                 the text column or embedding vector column.

    Returns:
        Dict[str, Any]: A dictionary containing extracted metadata.
    """
    metadata = {}

    if retriever_schema:
        for col, value in zip(columns, result):
            if col == "score" and include_score:
                metadata["score"] = value
            elif col == retriever_schema.doc_uri:
                metadata["doc_uri"] = value
            elif col == retriever_schema.primary_key:
                metadata["chunk_id"] = value
            elif col == "doc_uri" and retriever_schema.doc_uri:
                # Prioritize retriever_schema.doc_uri, don't override with the actual "doc_uri" column
                continue
            elif col == "chunk_id" and retriever_schema.primary_key:
                # Prioritize retriever_schema.primary_key, don't override with the actual "chunk_id" column
                continue
            elif col in ignore_cols:
                # ignore_cols has precedence over other_columns
                continue
            elif retriever_schema.other_columns is not None:
                if col in retriever_schema.other_columns:
                    metadata[col] = value
            else:
                metadata[col] = value
    else:
        for col, value in zip(columns, result):
            if col not in ignore_cols:
                metadata[col] = value
    return metadata


def parse_vector_search_response(
    search_resp: Dict,
    index_details: IndexDetails = None,  # deprecated
    text_column: str = None,  # deprecated
    *,
    retriever_schema: RetrieverSchema = None,
    ignore_cols: Optional[List[str]] = None,
    document_class: Any = dict,
    include_score: bool = False,
) -> List[Tuple[Dict, float]]:
    """
    Parse the search response into a list of Documents with score.
    The document_class parameter is used to specify the class of the document to be created.
    """
    if ignore_cols is None:
        ignore_cols = []

    if not include_score:
        ignore_cols.append("score")

    if retriever_schema:
        text_column = retriever_schema.text_column

    ignore_cols.append(text_column)

    columns = [col["name"] for col in search_resp.get("manifest", dict()).get("columns", [])]
    docs_with_score = []

    for result in search_resp.get("result", dict()).get("data_array", []):
        page_content = result[columns.index(text_column)]

        metadata = get_metadata(columns, result, retriever_schema, ignore_cols, include_score)

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
    columns: List[str],
    text_column: str,
    index_details: IndexDetails,
    doc_uri: str = None,
    primary_key: str = None,
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
    if doc_uri and doc_uri not in columns:
        columns.append(doc_uri)
    if primary_key and primary_key not in columns:
        columns.append(primary_key)

    # Validate specified columns are in the index
    if index_details.is_direct_access_index() and (index_schema := index_details.schema):
        if missing_columns := [c for c in columns if c not in index_schema]:
            raise ValueError(
                "Some columns specified in `columns` are not "
                f"in the index schema: {missing_columns}"
            )
    return columns
