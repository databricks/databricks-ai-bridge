from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_langchain import DatabricksMCPServer  # ty: ignore[unresolved-import]
from langchain_mcp_adapters.sessions import create_session  # ty: ignore[unresolved-import]

# BEGIN: mason add-sandbox
# The downscope is fixed when this file is generated. It is sent as MCP metadata,
# outside model-controlled tool arguments, on every sandbox call.
_SANDBOX_DOWNSCOPE: dict[str, list[dict[str, str]]] = {}  # __MASON_SANDBOX_DOWNSCOPE__


def _build_sandbox_mcp_server() -> DatabricksMCPServer:
    workspace_client = WorkspaceClient()
    return DatabricksMCPServer(
        name="system.ai.sandbox",
        url=(
            f"{workspace_client.config.host.rstrip('/')}/ai-gateway/mcp-services/system.ai.sandbox"
        ),
        workspace_client=workspace_client,
        timeout=120.0,
    )


async def _sandbox_tool_interceptor(request: Any, handler: Any) -> Any:
    if request.server_name != "system.ai.sandbox":
        return await handler(request)

    connection = _build_sandbox_mcp_server().to_connection_dict()
    async with create_session(connection) as session:
        await session.initialize()
        return await session.call_tool(
            request.name,
            request.args,
            meta={"downscope": _SANDBOX_DOWNSCOPE},
        )


# END: mason add-sandbox
