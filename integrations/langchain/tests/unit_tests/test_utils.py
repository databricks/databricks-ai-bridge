"""Test utilities module."""

import pytest
from unittest.mock import Mock, patch

from databricks_langchain.utils import get_openai_client


@pytest.mark.parametrize("kwargs,with_workspace_client", [
    ({"timeout": 45.0, "max_retries": 3}, True),
    ({"timeout": 30.0, "max_retries": 2}, False),
    ({}, True),
])
def test_get_openai_client(kwargs, with_workspace_client):
    mock_client = Mock()
    mock_workspace_client = Mock()

    with patch("databricks_langchain.utils.DatabricksOpenAI", return_value=mock_client) as mock_cls:
        with patch("databricks.sdk.WorkspaceClient", return_value=mock_workspace_client):
            wc = mock_workspace_client if with_workspace_client else None
            result = get_openai_client(workspace_client=wc, **kwargs)

    mock_cls.assert_called_once_with(workspace_client=mock_workspace_client, **kwargs)
    assert result == mock_client
