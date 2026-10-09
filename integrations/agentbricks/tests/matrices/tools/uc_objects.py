"""The UC objects the tools matrix uses, created fresh in each test's temporary schema."""

from __future__ import annotations

import dataclasses
import uuid

from common import MatrixError
from workspace_client import Workspace

NESTED_FUNCTION = "ab_nested_marker"
FUNCTION = "ab_uc_marker"
VOLUME = "ab_volume"
# The MCP server exposes a function as catalog__schema__function, capped at 64 characters.
MAX_TOOL_NAME_LENGTH = 64


@dataclasses.dataclass(frozen=True)
class UcFunction:
    """The scratch UC objects the tools use."""

    function: str
    # Called by ``function`` and deliberately never declared in agent.toml.
    nested_function: str
    volume: str
    volume_file_path: str
    volume_marker: str


def create_uc_function(workspace: Workspace, scratch_schema: str) -> UcFunction:
    """Create the marker functions and volume in ``scratch_schema``, whose deletion removes them."""
    catalog, _, schema = scratch_schema.partition(".")
    exposed_tool_name = f"{scratch_schema}.{FUNCTION}".replace(".", "__")
    if len(exposed_tool_name) > MAX_TOOL_NAME_LENGTH:
        raise MatrixError(
            f"The UC function's MCP tool name would exceed {MAX_TOOL_NAME_LENGTH} characters: "
            f"{exposed_tool_name!r}. Use a shorter catalog."
        )
    # The nested function first: the outer one resolves it when created.
    workspace.sql(
        f"CREATE FUNCTION `{catalog}`.`{schema}`.`{NESTED_FUNCTION}`"
        "(value STRING) RETURNS STRING "
        "COMMENT 'Transitive Agent Bricks E2E marker; never declared in agent.toml' "
        "RETURN concat('AGENTBRICKS_UC_OK:', value)"
    )
    workspace.sql(
        f"CREATE FUNCTION `{catalog}`.`{schema}`.`{FUNCTION}`"
        "(value STRING) RETURNS STRING "
        "COMMENT 'Deterministic Agent Bricks E2E marker tool' "
        f"RETURN `{catalog}`.`{schema}`.`{NESTED_FUNCTION}`(value)"
    )
    workspace.create_volume(catalog, schema, VOLUME)
    marker = f"AGENTBRICKS_VOLUME_{uuid.uuid4().hex}"
    file_path = f"/Volumes/{catalog}/{schema}/{VOLUME}/marker.txt"
    workspace.upload(file_path, marker.encode())
    return UcFunction(
        function=f"{scratch_schema}.{FUNCTION}",
        nested_function=f"{scratch_schema}.{NESTED_FUNCTION}",
        volume=f"{scratch_schema}.{VOLUME}",
        volume_file_path=file_path,
        volume_marker=marker,
    )
