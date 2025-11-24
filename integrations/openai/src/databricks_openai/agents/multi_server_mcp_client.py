from typing import List

from agents.mcp import MCPServerStreamableHttp


class DatabricksMultiServerMCPClient:
    def __init__(self, servers: List[MCPServerStreamableHttp]):
        self.servers = servers

    async def __aenter__(self):
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup_all()

    async def connect_all(self):
        for server in self.servers:
            await server.connect()

    async def cleanup_all(self):
        for server in self.servers:
            await server.cleanup()
