from databricks_agentkit.runtime import workspace


class _FakeWorkspaceClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_workspace_client_adds_account_routing_header(monkeypatch):
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", " 123456 ")
    monkeypatch.setattr(workspace, "WorkspaceClient", _FakeWorkspaceClient)

    client = workspace.workspace_client()

    assert client.kwargs == {
        "custom_headers": {"X-Databricks-Org-Id": "123456"},
    }
    assert workspace.workspace_headers() == {"X-Databricks-Org-Id": "123456"}


def test_workspace_client_uses_default_sdk_resolution_without_workspace_id(monkeypatch):
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    monkeypatch.setattr(workspace, "WorkspaceClient", _FakeWorkspaceClient)

    client = workspace.workspace_client()

    assert client.kwargs == {}
    assert workspace.workspace_headers() == {}


def test_mcp_headers_adds_opt_in_mas_traffic_routing(monkeypatch):
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "123456")
    monkeypatch.setenv("DATABRICKS_MAS_TRAFFIC_ID", "  testenv://liteswap/mas-sandbox-scope  ")

    assert workspace.mcp_headers() == {
        "X-Databricks-Org-Id": "123456",
        "x-databricks-traffic-id": "testenv://liteswap/mas-sandbox-scope",
    }


def test_mcp_headers_ignores_unrelated_environment(monkeypatch):
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("DATABRICKS_MAS_TRAFFIC_ID", "")
    monkeypatch.setenv("DATABRICKS_TRAFFIC_ID", "must-not-forward")

    assert workspace.mcp_headers() == {}
