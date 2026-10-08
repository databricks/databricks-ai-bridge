"""Unity Catalog access to MLflow Prompt Registry prompts an agent loads.

Databricks' Prompt Registry docs require "CREATE FUNCTION, EXECUTE, and MANAGE permissions to view or
create prompts", granted on the schema that holds them. A deployed agent that only *loads* its
prompts still needs all three, so `agentbricks deploy` grants them on each bound prompt's schema.
"""

from __future__ import annotations

_PERMISSIONS_PATH = "/api/2.1/unity-catalog/permissions"
SCHEMA_PRIVILEGES = ("USE_SCHEMA", "EXECUTE", "CREATE_FUNCTION", "MANAGE")


def schema_of(name: str) -> str:
    """``catalog.schema`` for prompt ``catalog.schema.name``."""
    catalog, schema, _ = name.split(".")
    return f"{catalog}.{schema}"


def grant_requests(schema: str, principal: str) -> list[tuple[str, dict, bool]]:
    """``(path, body, required)`` for letting ``principal`` load prompts in ``catalog.schema``.

    USE CATALOG is best-effort (the principal often holds it already); the schema grants are
    required.
    """
    catalog = schema.split(".")[0]
    return [
        (
            f"{_PERMISSIONS_PATH}/catalog/{catalog}",
            {"changes": [{"principal": principal, "add": ["USE_CATALOG"]}]},
            False,
        ),
        (
            f"{_PERMISSIONS_PATH}/schema/{schema}",
            {"changes": [{"principal": principal, "add": list(SCHEMA_PRIVILEGES)}]},
            True,
        ),
    ]
