from databricks_mason.runtime import workspace


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
