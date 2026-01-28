/**
 * MCP (Model Context Protocol) integration for Databricks.
 *
 * This module provides MCP client functionality with Databricks authentication,
 * allowing you to connect to MCP servers and use their tools with ChatDatabricks.
 *
 * @example
 * ```typescript
 * import { MultiServerMCPClient } from "@langchain/mcp-adapters";
 * import {
 *   ChatDatabricks,
 *   buildMCPServerConfig,
 *   DatabricksMCPServer,
 * } from "@databricks/langchainjs";
 *
 * // Build config with Databricks authentication
 * const mcpServers = await buildMCPServerConfig([
 *   new DatabricksMCPServer({
 *     name: "my-mcp-server",
 *     path: "/api/2.0/mcp/sql",
 *   }),
 * ]);
 *
 * // Create client and get tools
 * const client = new MultiServerMCPClient({ mcpServers });
 * const tools = await client.getTools();
 *
 * // Bind to chat model
 * const llm = new ChatDatabricks({ model: "my-model" });
 * const modelWithTools = model.bindTools(tools);
 *
 * // Clean up when done
 * await client.close();
 * ```
 */

export * from "./types.js";
export { MCPServer, DatabricksMCPServer } from "./mcp_server.js";
export { buildMCPServerConfig, type ServerInstance } from "./databricks_mcp_client.js";
export { DatabricksOAuthClientProvider } from "./databricks_oauth_provider.js";
