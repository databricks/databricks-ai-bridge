import pytest
from langchain_core.documents import Document

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
    page_content,
    column_1="column_1",
    column_2="column_2",
    column_3="column_3",
    column_4="column_4",
    document_class=dict,
):
    """
    Constructs a list of documents with associated scores for a search response,
    using a provided document class (e.g., dict or a custom class).

    Args:
        page_content (str): The name of the column whose value should be mapped to the "page_content" field
            in the constructed document. Must match one of the column names.
        column_1 (str or None): Optional. If None, this column is excluded from the output. Otherwise,
            includes this column in the output documents under the specified key. Defaults to "column_1".
        column_2 (str or None): Optional. Same behavior as column_1.
        column_3 (str or None): Optional. Same behavior as above.
        column_4 (str or None): Optional. Same behavior as above.
        document_class (type): The class to use when constructing the document objects. Typically `dict`,
            but can be a custom class. Defaults to `dict`.

    Returns:
        List[document_class]: A list of constructed document objects containing the selected columns,
        with one of them mapped as "page_content".
    """

    return [
        (
            document_class(
                page_content=f"row 1, column {page_content[-1]}",
                metadata={
                    **({column_1: "row 1, column 1"} if column_1 else {}),
                    **({column_2: "row 1, column 2"} if column_2 else {}),
                    **({column_3: 100} if column_3 else {}),
                    **({column_4: 5.8} if column_4 else {}),
                },
            ),
            0.673,
        ),
        (
            document_class(
                page_content=f"row 2, column {page_content[-1]}",
                metadata={
                    **({column_1: "row 2, column 1"} if column_1 else {}),
                    **({column_2: "row 2, column 2"} if column_2 else {}),
                    **({column_3: 200} if column_3 else {}),
                    **({column_4: 4.1} if column_4 else {}),
                },
            ),
            0.236,
        ),
    ]


def generate_parse_vector_search_response_test_cases():
    test_cases = []
    for document_class in [dict, Document]:
        test_cases.extend(
            [
                (  # Simple test case, only setting text_column
                    document_class,
                    RetrieverSchema(text_column="column_1"),
                    None,
                    construct_docs_with_score(
                        page_content="column_1", column_1=None, document_class=document_class
                    ),
                ),
                (  # Ensure that "ignore_cols" works
                    document_class,
                    RetrieverSchema(text_column="column_1"),
                    ["column_3"],
                    construct_docs_with_score(
                        page_content="column_1",
                        column_1=None,
                        column_3=None,
                        document_class=document_class,
                    ),
                ),
                (  # ignore_cols takes precedence over other_cols
                    document_class,
                    RetrieverSchema(text_column="column_1", other_columns=["column_3", "column_4"]),
                    ["column_3"],
                    construct_docs_with_score(
                        page_content="column_1",
                        column_1=None,
                        column_2=None,
                        column_3=None,
                        document_class=document_class,
                    ),
                ),
                (  # page_content takes precedence over other_cols (shouldn't be included in metadata)
                    document_class,
                    RetrieverSchema(text_column="column_1", other_columns=["column_1"]),
                    None,
                    construct_docs_with_score(
                        page_content="column_1",
                        column_1=None,
                        column_2=None,
                        column_3=None,
                        column_4=None,
                        document_class=document_class,
                    ),
                ),
                (  # Test mapping doc_uri and chunk_id
                    document_class,
                    RetrieverSchema(
                        text_column="column_1", doc_uri="column_2", chunk_id="column_3"
                    ),
                    None,
                    construct_docs_with_score(
                        page_content="column_1",
                        column_1=None,
                        column_2="doc_uri",
                        column_3="chunk_id",
                        document_class=document_class,
                    ),
                ),
                (  # doc_uri and chunk_id takes precendence over ignore_cols
                    document_class,
                    RetrieverSchema(
                        text_column="column_2", doc_uri="column_1", chunk_id="column_3"
                    ),
                    ["column_1", "column_3"],
                    construct_docs_with_score(
                        page_content="column_2",
                        column_1="doc_uri",
                        column_2=None,
                        column_3="chunk_id",
                        document_class=document_class,
                    ),
                ),
            ]
        )
    return test_cases


@pytest.mark.parametrize(
    "document_class,retriever_schema,ignore_cols,docs_with_score",
    generate_parse_vector_search_response_test_cases(),
)
def test_parse_vector_search_response(
    retriever_schema, ignore_cols, document_class, docs_with_score
):
    assert (
        parse_vector_search_response(search_resp, retriever_schema, ignore_cols, document_class)
        == docs_with_score
    )
