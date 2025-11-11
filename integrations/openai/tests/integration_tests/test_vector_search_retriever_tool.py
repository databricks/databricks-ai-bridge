from databricks.sdk import WorkspaceClient

from databricks_openai.vector_search_retriever_tool import VectorSearchRetrieverTool


def test_vs_tool_with_workspace_client():
    # tested manually with SP creds and PAT creds
    w = WorkspaceClient()
    vs_tool = VectorSearchRetrieverTool(index_name="main.default.cities_index", workspace_client=w)
    index = vs_tool._index
    assert index is not None
    if w.config.auth_type == "pat":
        assert index.personal_access_token is not None
    elif w.config.auth_type == "oauth-m2m":
        assert index.service_principal_client_id is not None
        assert index.service_principal_client_secret is not None
    else:
        raise ValueError(f"Unsupported auth type: {w.config.auth_type}")
