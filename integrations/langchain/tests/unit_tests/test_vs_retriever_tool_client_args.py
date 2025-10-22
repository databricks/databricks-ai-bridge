# integrations/langchain/tests/unit_tests/test_vs_retriever_tool_client_args.py

from typing import List
from langchain_core.embeddings import Embeddings

def test_client_args_pass_through(monkeypatch):
    captured = {}

    class FakeVSClient:
        def __init__(self, **client_kwargs):
            captured["kwargs"] = client_kwargs

        def get_index(self, **_):
            # Return a fake index object with a minimal describe() the tool expects
            class _Idx:
                def describe(self):
                    return {
                        "name": "catalog.schema.index",
                        "endpoint_name": "vs_endpoint",
                        "index_type": "DELTA_SYNC",  # validator sees delta-sync
                        "primary_key": "id",
                        "status": {"status": "ONLINE"},
                        # (No need to declare managed vs self-managed if we supply embedding+text_column)
                    }
            return _Idx()

    # Patch the canonical SDK client path
    monkeypatch.setattr(
        "databricks.vector_search.client.VectorSearchClient",
        FakeVSClient,
        raising=True,
    )

    # Minimal embeddings stub to satisfy self-managed requirement
    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[0.0] * 3 for _ in texts]
        def embed_query(self, text: str) -> List[float]:
            return [0.0, 0.0, 0.0]

    from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool

    tool = VectorSearchRetrieverTool(
        index_name="catalog.schema.index",
        text_column="body",               # required with self-managed/delta-sync
        embedding=DummyEmbeddings(),      # satisfy validator
        client_args={
            "service_principal_client_id": "abc",
            "service_principal_client_secret": "xyz",
            "disable_notice": True,
        },
    )

    assert captured["kwargs"]["service_principal_client_id"] == "abc"
    assert captured["kwargs"]["service_principal_client_secret"] == "xyz"
    assert captured["kwargs"]["disable_notice"] is True
