/**
 * Helper utilities for building MCP client configurations with Databricks authentication.
 */

import type { StreamableHTTPConnection } from "@langchain/mcp-adapters";
import { MCPServer, DatabricksMCPServer } from "./mcp_server.js";

export type ServerInstance = MCPServer | DatabricksMCPServer;

/**
 * Build MCP server configuration from an array of server instances.
 *
 * This helper converts `MCPServer` and `DatabricksMCPServer` instances into
 * the configuration format expected by `@langchain/mcp-adapters` `MultiServerMCPClient`.
 *
 * For `DatabricksMCPServer` instances, this resolves the URL from the Databricks SDK
 * config and adds authentication headers.
 *
 * @param servers - Array of MCPServer or DatabricksMCPServer instances
 * @returns Configuration object for MultiServerMCPClient's `mcpServers` option
 *
 * @example
 * ```typescript
 * import { MultiServerMCPClient } from "@langchain/mcp-adapters";
 * import {
 *   buildMCPServerConfig,
 *   DatabricksMCPServer,
 *   MCPServer,
 * } from "@databricks/langchainjs";
 *
 * // Create server instances
 * const servers = [
 *   new DatabricksMCPServer({
 *     name: "databricks-sql",
 *     path: "/api/2.0/mcp/sql",
 *   }),
 *   // Generic MCP server with custom auth
 *   new MCPServer({
 *     name: "other",
 *     url: "https://other-server.com/mcp",
 *     headers: { "Authorization": "Bearer token" },
 *   }),
 * ];
 *
 * // Build config and create client
 * const mcpServers = await buildMCPServerConfig(servers);
 * const client = new MultiServerMCPClient({ mcpServers });
 *
 * // Use the client
 * const tools = await client.getTools();
 *
 * // Clean up
 * await client.close();
 * ```
 */
export async function buildMCPServerConfig(
  servers: ServerInstance[]
): Promise<Record<string, StreamableHTTPConnection>> {
  const config: Record<string, StreamableHTTPConnection> = {};

  for (const server of servers) {
    config[server.name] = await server.toConnectionConfig();
  }

  return config;
}
