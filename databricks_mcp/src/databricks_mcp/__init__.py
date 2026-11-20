"""Public exports loaded on first access."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_mcp.connector import register_mcp_server_via_dcr
    from databricks_mcp.mcp import DatabricksMCPClient
    from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider

__all__ = [
    "DatabricksOAuthClientProvider",
    "DatabricksMCPClient",
    "register_mcp_server_via_dcr",
]

_MODULE_BY_NAME = {
    "DatabricksOAuthClientProvider": "databricks_mcp.oauth_provider",
    "DatabricksMCPClient": "databricks_mcp.mcp",
    "register_mcp_server_via_dcr": "databricks_mcp.connector",
}


def __getattr__(name: str) -> object:
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
