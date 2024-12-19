from typing import Any, List, Union
from urllib.parse import urlparse

import numpy as np
from enum import Enum
import json

from typing import (
    Dict,
    Optional
)

def get_deployment_client(target_uri: str) -> Any:
    if (target_uri != "databricks") and (urlparse(target_uri).scheme != "databricks"):
        raise ValueError("Invalid target URI. The target URI must be a valid databricks URI.")

    try:
        from mlflow.deployments import get_deploy_client  # type: ignore[import-untyped]

        return get_deploy_client(target_uri)
    except ImportError as e:
        raise ImportError(
            "Failed to create the client. "
            "Please run `pip install mlflow` to install "
            "required dependencies."
        ) from e


# Utility function for Maximal Marginal Relevance (MMR) reranking.
# Copied from langchain_community/vectorstores/utils.py to avoid cross-dependency
Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: Query embedding.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
        k: Number of Documents to return. Defaults to 4.

    Returns:
        List of indices of embeddings selected by maximal marginal relevance.
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Number of columns in X and Y must be the same. X has shape"
            f"{X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity

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
