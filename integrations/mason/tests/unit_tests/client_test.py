"""Tests for the public typed Mason client facade."""

from unittest import mock

from resource_test_fixtures import resource_client

from databricks_mason import DatabricksAgentClient


def test_constructs_agent_api_client_from_profile() -> None:
    with mock.patch("databricks_mason.client.AgentApiClient") as constructor:
        client = DatabricksAgentClient(profile="p")

    constructor.assert_called_once_with(profile="p")
    assert client.memory_stores._api is constructor.return_value
    assert client.session_stores._api is constructor.return_value


def test_injected_api_client_is_shared() -> None:
    client, api = resource_client()

    assert client.memory_stores._api is api
    assert client.session_stores._api is api
