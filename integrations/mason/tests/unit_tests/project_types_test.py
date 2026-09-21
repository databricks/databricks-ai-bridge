"""Selection enums retain Mason's lowercase command and manifest values."""

import json

import pytest

from databricks_mason.errors import AgentCliError
from databricks_mason.project_types import (
    AgentFramework,
    AgentServer,
    parse_framework,
    parse_server,
)


@pytest.mark.parametrize("framework", list(AgentFramework))
def test_framework_parser_accepts_enum_and_value(framework: AgentFramework):
    assert parse_framework(framework) is framework
    assert parse_framework(framework.value) is framework


@pytest.mark.parametrize("server", list(AgentServer))
def test_server_parser_accepts_enum_and_value(server: AgentServer):
    assert parse_server(server) is server
    assert parse_server(server.value) is server


@pytest.mark.parametrize("selection", [*AgentFramework, *AgentServer])
def test_enum_text_and_json_use_the_lowercase_value(selection: AgentFramework | AgentServer):
    assert str(selection) == selection.value
    assert f"{selection}" == selection.value
    assert json.loads(json.dumps(selection)) == selection.value


@pytest.mark.parametrize("parser", [parse_framework, parse_server])
@pytest.mark.parametrize("invalid", [None, 42, [], {}, "", "unknown"])
def test_selection_parsers_surface_cli_errors(parser, invalid: object):
    with pytest.raises(AgentCliError) as error:
        parser(invalid)
    assert "Unsupported Mason" in error.value.message
    assert "Supported" in (error.value.hint or "")
    assert "AgentFramework" not in str(error.value)
    assert "AgentServer" not in str(error.value)


def test_invalid_cross_enum_value_does_not_leak_enum_representation():
    with pytest.raises(AgentCliError) as error:
        parse_framework(AgentServer.MASON)
    assert error.value.message == "Unsupported Mason framework 'mason'."
