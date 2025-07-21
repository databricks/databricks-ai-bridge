from unittest.mock import MagicMock, patch

import pytest

from databricks_dspy.clients.databricks_lm import DatabricksLM


def test_forward_invokes_authenticate():
    mock_ws_client = MagicMock()
    mock_ws_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
    mock_ws_client.config.host = "https://test-host"
    mock_ws_client.current_user.me.return_value = "some valid value"
    lm = DatabricksLM(model="databricks/test-model", workspace_client=mock_ws_client)

    with patch("databricks_dspy.clients.databricks_lm.dspy.LM.forward") as mock_super_forward:
        # Call the LM (`DatabricksLM.__call__` will call `forward`)
        lm("test input")
        # `authenticate` should be called
        assert mock_ws_client.config.authenticate.called
        mock_super_forward.assert_called_once()

        _, kwargs = mock_super_forward.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer token"}
        assert kwargs["api_base"] == "https://test-host/serving-endpoints"
        assert kwargs["prompt"] == "test input"


def test_valid_credentials():
    with patch("databricks_dspy.clients.databricks_lm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_current_user = MagicMock()
        # Simulate valid credentials
        mock_current_user.me.return_value = "some valid value"
        mock_ws.current_user = mock_current_user
        MockWSClient.return_value = mock_ws

        DatabricksLM(model="databricks/test-model")
        mock_current_user.me.assert_called_once()


def test_invalid_credentials_raise_error():
    with patch("databricks_dspy.clients.databricks_lm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_current_user = MagicMock()
        # Simulate invalid credentials (raise on me())
        mock_current_user.me.side_effect = Exception("auth failed")
        mock_ws.current_user = mock_current_user
        MockWSClient.return_value = mock_ws

        with pytest.raises(RuntimeError, match="Failed to validate databricks credentials"):
            DatabricksLM(model="databricks/test-model")
        mock_current_user.me.assert_called_once()


def test_create_pt_endpoint_failed():
    with patch("databricks_dspy.clients.databricks_lm.WorkspaceClient") as MockWSClient:  # noqa: E501
        mock_ws = MagicMock()
        # Simulate endpoint creation failure.
        mock_ws.serving_endpoints.create_provisioned_throughput_endpoint_and_wait.side_effect = (
            Exception("PT endpoint creation failed")
        )  # noqa: E501
        MockWSClient.return_value = mock_ws

        pt_entity = MagicMock()
        with pytest.raises(RuntimeError) as e:
            DatabricksLM(
                model="databricks/test-model", create_pt_endpoint=True, pt_entity=pt_entity
            )
        assert "Failed to create provisioned throughput endpoint" in str(e)
        assert "`create_pt_endpoint=True` is only supported in Databricks notebooks now." in str(e)
