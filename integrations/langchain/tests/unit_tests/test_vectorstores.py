import uuid
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest
from databricks.vector_search.client import VectorSearchIndex  # type: ignore
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    DIRECT_ACCESS_INDEX,
    ENDPOINT_NAME,
    INDEX_DETAILS,
    INPUT_TEXTS,
    mock_vs_client,  # noqa: F401
)

from databricks_langchain.vectorstores import DatabricksVectorSearch
from tests.utils.vector_search import (
    EMBEDDING_MODEL,
    FakeEmbeddings,
)


def init_vector_search(
    index_name: str, columns: Optional[List[str]] = None
) -> DatabricksVectorSearch:
    kwargs: Dict[str, Any] = {
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
    return DatabricksVectorSearch(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_init(index_name: str) -> None:
    vectorsearch = init_vector_search(index_name)
    assert vectorsearch.index.describe() == INDEX_DETAILS[index_name]


def test_init_with_endpoint_name() -> None:
    vectorsearch = DatabricksVectorSearch(
        endpoint=ENDPOINT_NAME,
        index_name=DELTA_SYNC_INDEX,
    )
    assert vectorsearch.index.describe() == INDEX_DETAILS[DELTA_SYNC_INDEX]


@pytest.mark.parametrize(
    "index_name", [None, 123, "invalid.name", MagicMock(spec=VectorSearchIndex)]
)
def test_init_fail_invalid_index_name(index_name) -> None:
    with pytest.raises(ValueError, match="The `index_name` parameter must be"):
        DatabricksVectorSearch(index_name=index_name)


def test_init_fail_text_column_mismatch() -> None:
    with pytest.raises(ValueError, match=f"The index '{DELTA_SYNC_INDEX}' has"):
        DatabricksVectorSearch(
            index_name=DELTA_SYNC_INDEX,
            text_column="some_other_column",
        )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DELTA_SYNC_INDEX})
def test_init_fail_no_text_column(index_name: str) -> None:
    with pytest.raises(ValueError, match="The `text_column` parameter is required"):
        DatabricksVectorSearch(
            index_name=index_name,
            embedding=EMBEDDING_MODEL,
        )


def test_init_fail_columns_not_in_schema() -> None:
    columns = ["some_random_column"]
    with pytest.raises(ValueError, match="Some columns specified in `columns`"):
        init_vector_search(DIRECT_ACCESS_INDEX, columns=columns)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DELTA_SYNC_INDEX})
def test_init_fail_no_embedding(index_name: str) -> None:
    with pytest.raises(ValueError, match="The `embedding` parameter is required"):
        DatabricksVectorSearch(
            index_name=index_name,
            text_column="text",
        )


def test_init_fail_embedding_already_specified_in_source() -> None:
    with pytest.raises(ValueError, match=f"The index '{DELTA_SYNC_INDEX}' uses"):
        DatabricksVectorSearch(
            index_name=DELTA_SYNC_INDEX,
            embedding=EMBEDDING_MODEL,
        )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DELTA_SYNC_INDEX})
def test_init_fail_embedding_dim_mismatch(index_name: str) -> None:
    with pytest.raises(ValueError, match="embedding model's dimension '1000' does not match"):
        DatabricksVectorSearch(
            index_name=index_name,
            text_column="text",
            embedding=FakeEmbeddings(1000),
        )


def test_from_texts_not_supported() -> None:
    with pytest.raises(NotImplementedError, match="`from_texts` is not supported"):
        DatabricksVectorSearch.from_texts(INPUT_TEXTS, EMBEDDING_MODEL)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DIRECT_ACCESS_INDEX})
def test_add_texts_not_supported_for_delta_sync_index(index_name: str) -> None:
    vectorsearch = init_vector_search(index_name)
    with pytest.raises(
        NotImplementedError,
        match="`add_texts` is only supported for direct-access index.",
    ):
        vectorsearch.add_texts(INPUT_TEXTS)


def is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


def test_add_texts() -> None:
    vectorsearch = init_vector_search(DIRECT_ACCESS_INDEX)
    ids = [idx for idx, i in enumerate(INPUT_TEXTS)]
    vectors = EMBEDDING_MODEL.embed_documents(INPUT_TEXTS)

    added_ids = vectorsearch.add_texts(INPUT_TEXTS, ids=ids)
    vectorsearch.index.upsert.assert_called_once_with(
        [
            {
                "id": id_,
                "text": text,
                "text_vector": vector,
            }
            for text, vector, id_ in zip(INPUT_TEXTS, vectors, ids)
        ]
    )
    assert len(added_ids) == len(INPUT_TEXTS)
    assert added_ids == ids


def test_add_texts_handle_single_text() -> None:
    vectorsearch = init_vector_search(DIRECT_ACCESS_INDEX)
    vectors = EMBEDDING_MODEL.embed_documents(INPUT_TEXTS)

    added_ids = vectorsearch.add_texts(INPUT_TEXTS[0])
    vectorsearch.index.upsert.assert_called_once_with(
        [
            {
                "id": id_,
                "text": text,
                "text_vector": vector,
            }
            for text, vector, id_ in zip(INPUT_TEXTS, vectors, added_ids)
        ]
    )
    assert len(added_ids) == 1
    assert is_valid_uuid(added_ids[0])


def test_add_texts_with_default_id() -> None:
    vectorsearch = init_vector_search(DIRECT_ACCESS_INDEX)
    vectors = EMBEDDING_MODEL.embed_documents(INPUT_TEXTS)

    added_ids = vectorsearch.add_texts(INPUT_TEXTS)
    vectorsearch.index.upsert.assert_called_once_with(
        [
            {
                "id": id_,
                "text": text,
                "text_vector": vector,
            }
            for text, vector, id_ in zip(INPUT_TEXTS, vectors, added_ids)
        ]
    )
    assert len(added_ids) == len(INPUT_TEXTS)
    assert all([is_valid_uuid(id_) for id_ in added_ids])


def test_add_texts_with_metadata() -> None:
    vectorsearch = init_vector_search(DIRECT_ACCESS_INDEX)
    vectors = EMBEDDING_MODEL.embed_documents(INPUT_TEXTS)
    metadatas = [{"feat1": str(i), "feat2": i + 1000} for i in range(len(INPUT_TEXTS))]

    added_ids = vectorsearch.add_texts(INPUT_TEXTS, metadatas=metadatas)
    vectorsearch.index.upsert.assert_called_once_with(
        [
            {
                "id": id_,
                "text": text,
                "text_vector": vector,
                **metadata,  # type: ignore[arg-type]
            }
            for text, vector, id_, metadata in zip(INPUT_TEXTS, vectors, added_ids, metadatas)
        ]
    )
    assert len(added_ids) == len(INPUT_TEXTS)
    assert all([is_valid_uuid(id_) for id_ in added_ids])


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DELTA_SYNC_INDEX})
def test_embeddings_property(index_name: str) -> None:
    vectorsearch = init_vector_search(index_name)
    assert vectorsearch.embeddings == EMBEDDING_MODEL


def test_delete() -> None:
    vectorsearch = init_vector_search(DIRECT_ACCESS_INDEX)
    vectorsearch.delete(["some id"])
    vectorsearch.index.delete.assert_called_once_with(["some id"])


def test_delete_fail_no_ids() -> None:
    vectorsearch = init_vector_search(DIRECT_ACCESS_INDEX)
    with pytest.raises(ValueError, match="ids must be provided."):
        vectorsearch.delete()


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DIRECT_ACCESS_INDEX})
def test_delete_not_supported_for_delta_sync_index(index_name: str) -> None:
    vectorsearch = init_vector_search(index_name)
    with pytest.raises(NotImplementedError, match="`delete` is only supported for direct-access"):
        vectorsearch.delete(["some id"])


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize("query_type", [None, "ANN"])
def test_similarity_search(index_name: str, query_type: Optional[str]) -> None:
    vectorsearch = init_vector_search(index_name)
    query = "foo"
    filters = {"some filter": True}
    limit = 7

    search_result = vectorsearch.similarity_search(
        query, k=limit, filter=filters, query_type=query_type
    )
    if index_name == DELTA_SYNC_INDEX:
        vectorsearch.index.similarity_search.assert_called_once_with(
            columns=["id", "text"],
            query_text=query,
            query_vector=None,
            filters=filters,
            num_results=limit,
            query_type=query_type,
        )
    else:
        vectorsearch.index.similarity_search.assert_called_once_with(
            columns=["id", "text"],
            query_text=None,
            query_vector=EMBEDDING_MODEL.embed_query(query),
            filters=filters,
            num_results=limit,
            query_type=query_type,
        )
    assert len(search_result) == len(INPUT_TEXTS)
    assert sorted([d.page_content for d in search_result]) == sorted(INPUT_TEXTS)
    assert all(["id" in d.metadata for d in search_result])


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_similarity_search_hybrid(index_name: str) -> None:
    vectorsearch = init_vector_search(index_name)
    query = "foo"
    filters = {"some filter": True}
    limit = 7

    search_result = vectorsearch.similarity_search(
        query, k=limit, filter=filters, query_type="HYBRID"
    )
    if index_name == DELTA_SYNC_INDEX:
        vectorsearch.index.similarity_search.assert_called_once_with(
            columns=["id", "text"],
            query_text=query,
            query_vector=None,
            filters=filters,
            num_results=limit,
            query_type="HYBRID",
        )
    else:
        vectorsearch.index.similarity_search.assert_called_once_with(
            columns=["id", "text"],
            query_text=query,
            query_vector=EMBEDDING_MODEL.embed_query(query),
            filters=filters,
            num_results=limit,
            query_type="HYBRID",
        )
    assert len(search_result) == len(INPUT_TEXTS)
    assert sorted([d.page_content for d in search_result]) == sorted(INPUT_TEXTS)
    assert all(["id" in d.metadata for d in search_result])


def test_similarity_search_passing_kwargs() -> None:
    vectorsearch = init_vector_search(DELTA_SYNC_INDEX)
    query = "foo"
    filters = {"some filter": True}
    query_type = "ANN"

    search_result = vectorsearch.similarity_search(
        query,
        k=5,
        filter=filters,
        query_type=query_type,
        score_threshold=0.5,
        num_results=10,
        random_parameters="not included",
    )
    vectorsearch.index.similarity_search.assert_called_once_with(
        columns=["id", "text"],
        query_text=query,
        query_vector=None,
        filters=filters,
        query_type=query_type,
        num_results=5,  # maintained
        score_threshold=0.5,  # passed
    )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DELTA_SYNC_INDEX})
@pytest.mark.parametrize(
    "columns, expected_columns",
    [
        (None, {"id"}),
        (["id", "text", "text_vector"], {"text_vector", "id"}),
    ],
)
def test_mmr_search(
    index_name: str, columns: Optional[List[str]], expected_columns: Set[str]
) -> None:
    vectorsearch = init_vector_search(index_name, columns=columns)

    query = INPUT_TEXTS[0]
    filters = {"some filter": True}
    limit = 1

    search_result = vectorsearch.max_marginal_relevance_search(query, k=limit, filters=filters)
    assert [doc.page_content for doc in search_result] == [INPUT_TEXTS[0]]
    assert [set(doc.metadata.keys()) for doc in search_result] == [expected_columns]


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DELTA_SYNC_INDEX})
def test_mmr_parameters(index_name: str) -> None:
    vectorsearch = init_vector_search(index_name)

    query = INPUT_TEXTS[0]
    limit = 1
    fetch_k = 3
    lambda_mult = 0.25
    filters = {"some filter": True}

    with patch("databricks_langchain.vectorstores.maximal_marginal_relevance") as mock_mmr:
        mock_mmr.return_value = [2]
        retriever = vectorsearch.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": limit,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                "filter": filters,
            },
        )
        search_result = retriever.invoke(query)

    mock_mmr.assert_called_once()
    assert mock_mmr.call_args[1]["lambda_mult"] == lambda_mult
    assert vectorsearch.index.similarity_search.call_args[1]["num_results"] == fetch_k
    assert vectorsearch.index.similarity_search.call_args[1]["filters"] == filters
    assert len(search_result) == limit


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize("threshold", [0.4, 0.5, 0.8])
def test_similarity_score_threshold(index_name: str, threshold: float) -> None:
    query = INPUT_TEXTS[0]
    limit = len(INPUT_TEXTS)

    vectorsearch = init_vector_search(index_name)
    retriever = vectorsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": limit, "score_threshold": threshold},
    )
    search_result = retriever.invoke(query)
    if threshold <= 0.5:
        assert len(search_result) == len(INPUT_TEXTS)
    else:
        assert len(search_result) == 0


def test_standard_params() -> None:
    vectorstore = init_vector_search(DIRECT_ACCESS_INDEX)
    retriever = vectorstore.as_retriever()
    ls_params = retriever._get_ls_params()
    assert ls_params == {
        "ls_retriever_name": "vectorstore",
        "ls_vector_store_provider": "DatabricksVectorSearch",
        "ls_embedding_provider": "FakeEmbeddings",
    }

    vectorstore = init_vector_search(DELTA_SYNC_INDEX)
    retriever = vectorstore.as_retriever()
    ls_params = retriever._get_ls_params()
    assert ls_params == {
        "ls_retriever_name": "vectorstore",
        "ls_vector_store_provider": "DatabricksVectorSearch",
    }


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DELTA_SYNC_INDEX})
@pytest.mark.parametrize("query_type", [None, "ANN"])
def test_similarity_search_by_vector(index_name: str, query_type: Optional[str]) -> None:
    vectorsearch = init_vector_search(index_name)
    query_embedding = EMBEDDING_MODEL.embed_query("foo")
    filters = {"some filter": True}
    limit = 7

    search_result = vectorsearch.similarity_search_by_vector(
        query_embedding, k=limit, filter=filters, query_type=query_type
    )
    vectorsearch.index.similarity_search.assert_called_once_with(
        columns=["id", "text"],
        query_vector=query_embedding,
        filters=filters,
        num_results=limit,
        query_type=query_type,
        query_text=None,
    )
    assert len(search_result) == len(INPUT_TEXTS)
    assert sorted([d.page_content for d in search_result]) == sorted(INPUT_TEXTS)
    assert all(["id" in d.metadata for d in search_result])


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES - {DELTA_SYNC_INDEX})
def test_similarity_search_by_vector_hybrid(index_name: str) -> None:
    vectorsearch = init_vector_search(index_name)
    query_embedding = EMBEDDING_MODEL.embed_query("foo")
    filters = {"some filter": True}
    limit = 7

    search_result = vectorsearch.similarity_search_by_vector(
        query_embedding, k=limit, filter=filters, query_type="HYBRID", query="foo"
    )
    vectorsearch.index.similarity_search.assert_called_once_with(
        columns=["id", "text"],
        query_vector=query_embedding,
        filters=filters,
        num_results=limit,
        query_type="HYBRID",
        query_text="foo",
    )
    assert len(search_result) == len(INPUT_TEXTS)
    assert sorted([d.page_content for d in search_result]) == sorted(INPUT_TEXTS)
    assert all(["id" in d.metadata for d in search_result])


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_similarity_search_empty_result(index_name: str) -> None:
    vectorsearch = init_vector_search(index_name)
    vectorsearch.index.similarity_search.return_value = {
        "manifest": {
            "column_count": 3,
            "columns": [
                {"name": "id"},
                {"name": "text"},
                {"name": "score"},
            ],
        },
        "result": {
            "row_count": 0,
            "data_array": [],
        },
        "next_page_token": "",
    }

    search_result = vectorsearch.similarity_search("foo")
    assert len(search_result) == 0


def test_similarity_search_by_vector_not_supported_for_managed_embedding() -> None:
    vectorsearch = init_vector_search(DELTA_SYNC_INDEX)
    query_embedding = EMBEDDING_MODEL.embed_query("foo")
    filters = {"some filter": True}
    limit = 7

    with pytest.raises(NotImplementedError, match="`similarity_search_by_vector` is not supported"):
        vectorsearch.similarity_search_by_vector(query_embedding, k=limit, filters=filters)
