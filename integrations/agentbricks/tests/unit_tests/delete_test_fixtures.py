"""A fake SDK workspace client for `deployments delete` tests."""

from typing import Optional
from unittest import mock

from databricks.sdk.errors import NotFound
from databricks.sdk.service.apps import App, AppDeployment


def fake_workspace_client(*, source_code_path: Optional[str] = None) -> mock.Mock:
    """An SDK client whose App exists until ``remove_app()``; its folder deletes succeed.

    A mock, so tests can assert the order of `apps.get` and `workspace.delete` calls.
    """
    client = mock.Mock()
    exists = True

    def get(name: str) -> App:
        if not exists:
            raise NotFound(f"App with name {name} does not exist or is deleted.")
        deployment = AppDeployment(source_code_path=source_code_path) if source_code_path else None
        return App(name=name, active_deployment=deployment)

    def remove_app() -> None:
        nonlocal exists
        exists = False

    client.apps.get.side_effect = get
    client.remove_app = remove_app
    return client
