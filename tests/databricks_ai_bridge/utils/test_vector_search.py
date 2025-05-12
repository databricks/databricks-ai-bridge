import pytest

from databricks_ai_bridge.utils.vector_search import RetrieverSchema, parse_vector_search_response

search_resp = {
    "manifest": {
        "column_count": 2,
        "columns": [{"name": f"column_{i}"} for i in range(1, 5)] + [{"name": "score"}],
    },
    "result": {
        "data_array": [
            ["row 1, column 1", "row 1, column 2", 100, 5.8, 0.673],
            ["row 2, column 1", "row 2, column 2", 200, 4.1, 0.236],
        ]
    },
}


def construct_docs_with_score(
    page_content_column: str,
    column_1: str = None,
    column_2: str = None,
    column_3: str = None,
    column_4: str = None,
    document_class=dict,
):
    """
    Constructs a list of (document, score) tuples based on simulated search response data.

    Args:
        page_content_column: The name of the column to use for page_content.
        column_1, column_2, column_3, column_4: Column names to include in metadata. Excluded if set to None.
        document_class: Type used to construct each document (e.g., dict or custom class).

    Returns:
        List of (document_class, score) tuples.
    """

    def make_document(row_index: int, score: float):
        num_cols = [[100, 5.8], [200, 4.1]]

        metadata = {}
        if column_1:
            metadata[column_1] = f"row {row_index+1}, column 1"
        if column_2:
            metadata[column_2] = f"row {row_index+1}, column 2"
        if column_3:
            metadata[column_3] = num_cols[row_index][0]
        if column_4:
            metadata[column_4] = num_cols[row_index][1]

        page_content = f"row {row_index+1}, column {page_content_column[-1]}"
        return (document_class(page_content=page_content, metadata=metadata), score)

    return [make_document(0, 0.673), make_document(1, 0.236)]


@pytest.mark.parametrize(
    "retriever_schema,ignore_cols,docs_with_score",
    [
        (  # Simple test case, only setting text_column
            RetrieverSchema(text_column="column_1"),
            None,
            construct_docs_with_score(
                page_content_column="column_1",
                column_2="column_2",
                column_3="column_3",
                column_4="column_4",
            ),
        ),
        (  # Ensure that "ignore_cols" works
            RetrieverSchema(text_column="column_1"),
            ["column_3"],
            construct_docs_with_score(
                page_content_column="column_1",
                column_2="column_2",
                column_4="column_4",
            ),
        ),
        (  # ignore_cols takes precedence over other_cols
            RetrieverSchema(text_column="column_1", other_columns=["column_3", "column_4"]),
            ["column_3"],
            construct_docs_with_score(page_content_column="column_1", column_4="column_4"),
        ),
        (  # page_content takes precedence over other_cols (shouldn't be included in metadata)
            RetrieverSchema(text_column="column_1", other_columns=["column_1"]),
            None,
            construct_docs_with_score(page_content_column="column_1"),
        ),
        (  # Test mapping doc_uri and primary_key
            RetrieverSchema(text_column="column_1", doc_uri="column_2", primary_key="column_3"),
            None,
            construct_docs_with_score(
                page_content_column="column_1",
                column_2="doc_uri",
                column_3="chunk_id",
                column_4="column_4",
            ),
        ),
        (  # doc_uri and primary_key takes precendence over ignore_cols
            RetrieverSchema(text_column="column_2", doc_uri="column_1", primary_key="column_3"),
            ["column_1", "column_3"],
            construct_docs_with_score(
                page_content_column="column_2",
                column_1="doc_uri",
                column_3="chunk_id",
                column_4="column_4",
            ),
        ),
    ],
)
def test_parse_vector_search_response(retriever_schema, ignore_cols, docs_with_score):
    assert (
        parse_vector_search_response(
            search_resp, retriever_schema=retriever_schema, ignore_cols=ignore_cols
        )
        == docs_with_score
    )


def test_parse_vector_search_response_without_retriever_schema():
    assert (
        parse_vector_search_response(search_resp, text_column="column_1", ignore_cols=["column_2"])
        == construct_docs_with_score(
            page_content_column="column_2",
            column_3="column_3",
            column_4="column_4",
        ),
    )
