"""POST-only JSON-RPC MCP client for external/managed UC connections.

External MCP connections (``/api/2.0/mcp/external/<connection>``) are fronted by a
POST-only JSON-REST bridge: it rejects any non-POST method with ``405 Method Not
Allowed`` and replies with ``application/json`` (never ``text/event-stream``). It is a
request/response transcoder with no GET endpoint, no SSE stream, and no server-initiated
messages.

The standard Streamable HTTP client (``mcp.client.streamable_http.streamablehttp_client``,
used by :class:`~databricks_mcp.mcp.DatabricksMCPClient` for managed servers) opens a
background GET SSE stream for server -> client messages. Against this POST-only bridge that
GET is rejected with 405 and the stream task dies; a later request whose response handling
expects coordination with that (now-dead) stream then blocks until timeout. See
modelcontextprotocol/python-sdk#1941 ("hangs indefinitely when connecting to POST-only MCP
servers").

This client speaks plain request/response JSON-RPC over POST instead, which is exactly what
the bridge supports: ``initialize`` -> ``notifications/initialized`` -> ``tools/list`` /
``tools/call``. Responses are parsed as JSON, or, defensively, as a single ``data:`` frame if
the server ever replies with a one-shot ``text/event-stream`` body.
"""

import json
from typing import Any

import httpx
from databricks.sdk import WorkspaceClient
from mcp.types import CallToolResult, Tool

# The bridge negotiates the 2025-03-26 Streamable HTTP spec. Advertise it explicitly rather
# than the SDK's LATEST_PROTOCOL_VERSION, which can be newer than the bridge supports.
_PROTOCOL_VERSION = "2025-03-26"
_DEFAULT_TIMEOUT_SECONDS = 30.0


class PostOnlyMCPClient:
    """Minimal request/response JSON-RPC MCP client over HTTP POST.

    Returns the same ``mcp.types`` objects as the Streamable HTTP path so it is a drop-in
    for the external-connection branch of :class:`DatabricksMCPClient`.
    """

    def __init__(
        self,
        server_url: str,
        workspace_client: WorkspaceClient,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.server_url = server_url
        self.workspace_client = workspace_client
        self._session_id: str | None = None
        self._next_id = 1
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    async def __aenter__(self) -> "PostOnlyMCPClient":
        await self._initialize()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        # Resolve a fresh bearer token from the workspace client on each handshake, matching
        # how DatabricksMCPClient authenticates elsewhere.
        authorization = self.workspace_client.config.authenticate()["Authorization"]
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": _PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    @staticmethod
    def _parse_body(response: httpx.Response) -> dict[str, Any]:
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if payload:
                        return json.loads(payload)
            raise RuntimeError(
                f"No JSON-RPC frame in event-stream response: {response.text[:300]!r}"
            )
        return response.json()

    async def _send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        is_notification: bool = False,
    ) -> dict[str, Any] | None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not is_notification:
            body["id"] = self._next_id
            self._next_id += 1
        if params is not None:
            body["params"] = params

        response = await self._client.post(self.server_url, headers=self._headers(), json=body)

        if method == "initialize":
            self._session_id = response.headers.get("Mcp-Session-Id")

        response.raise_for_status()
        if is_notification:
            return None

        frame = self._parse_body(response)
        if "error" in frame:
            raise RuntimeError(f"MCP {method} error: {frame['error']}")
        return frame.get("result", {})

    async def _initialize(self) -> None:
        await self._send(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "databricks-mcp", "version": "1"},
            },
        )
        # The MCP spec requires this notification before the server answers tools/* requests.
        await self._send("notifications/initialized", is_notification=True)

    async def list_tools(self) -> list[Tool]:
        result = await self._send("tools/list", {})
        return [Tool.model_validate(tool) for tool in (result or {}).get("tools", [])]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        result = await self._send(
            "tools/call", {"name": tool_name, "arguments": arguments or {}}
        )
        return CallToolResult.model_validate(result)
