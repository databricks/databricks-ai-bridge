"""
This file contains the integration test for DatabricksEmbeddings class.

We run the integration tests nightly by the trusted CI/CD system defined in
a private repository, in order to securely run the tests. With this design,
integration test is not intended to be run manually by OSS contributors.
If you want to update the DatabricksEmbeddings implementation and you think
that you need to update the corresponding integration test, please contact to
the maintainers of the repository to verify the changes.
"""

from langchain_databricks import DatabricksEmbeddings

_TEST_ENDPOINT = "databricks-bge-large-en"


def test_embedding_documents() -> None:
    documents = ["foo bar"]
    embedding = DatabricksEmbeddings(endpoint=_TEST_ENDPOINT)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_embedding_query() -> None:
    document = "foo bar"
    embedding = DatabricksEmbeddings(endpoint=_TEST_ENDPOINT)
    output = embedding.embed_query(document)
    assert len(output) > 0
