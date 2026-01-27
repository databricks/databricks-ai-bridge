/**
 * MCP (Model Context Protocol) integration for Databricks.
 *
 * This module provides MCP client functionality with Databricks authentication,
 * allowing you to connect to MCP servers and use their tools with ChatDatabricks.
 *
 * @example
 * ```typescript
 * import {
 *   ChatDatabricks,
 *   DatabricksMultiServerMCPClient,
 *   DatabricksMCPServer,
 * } from "@databricks/langchainjs";
 *
 * // Create MCP client with Databricks authentication
 * const mcpClient = new DatabricksMultiServerMCPClient([
 *   new DatabricksMCPServer({
 *     name: "my-mcp-server",
 *     path: "/api/2.0/mcp/sql",
 *   }),
 * ]);
 *
 * // Get tools and bind to chat model
 * const tools = await mcpClient.getTools();
 * const model = new ChatDatabricks({ endpoint: "my-endpoint" });
 * const modelWithTools = model.bindTools(tools);
 *
 * // Clean up when done
 * await mcpClient.close();
 * ```
 */

export * from "./types.js";
export { MCPServer, DatabricksMCPServer } from "./mcp_server.js";
export { DatabricksMultiServerMCPClient } from "./databricks_mcp_client.js";
export { DatabricksOAuthClientProvider } from "./databricks_oauth_provider.js";
