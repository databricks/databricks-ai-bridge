/**
 * DatabricksMultiServerMCPClient - MCP client with Databricks authentication support
 *
 * Wraps @langchain/mcp-adapters MultiServerMCPClient with simplified configuration
 * and Databricks-specific OAuth authentication.
 */

import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import type { StreamableHTTPConnection } from "@langchain/mcp-adapters";
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { DatabricksMCPClientOptions } from "./types.js";
import { MCPServer, DatabricksMCPServer } from "./mcp_server.js";

export type ServerInstance = MCPServer | DatabricksMCPServer;

/**
 * MCP client with simplified configuration for Databricks servers.
 *
 * This wrapper provides an ergonomic interface for connecting to multiple MCP servers,
 * with special support for Databricks OAuth authentication. It wraps the standard
 * @langchain/mcp-adapters MultiServerMCPClient with:
 *
 * - Simplified server configuration through MCPServer/DatabricksMCPServer classes
 * - Automatic OAuth token management for Databricks servers
 * - Per-server error handling configuration applied to tools
 * - Parallel tool loading for improved performance
 *
 * @example
 * ```typescript
 * import {
 *   DatabricksMultiServerMCPClient,
 *   DatabricksMCPServer,
 *   MCPServer
 * } from "@databricks/langchain-ts";
 *
 * // Use the static create method for async initialization
 * const client = await DatabricksMultiServerMCPClient.create([
 *   // Databricks server with automatic OAuth
 *   new DatabricksMCPServer({
 *     name: "databricks",
 *     url: "https://workspace.databricks.com/api/mcp",
 *     handleToolError: true,
 *   }),
 *   // Generic MCP server with custom auth
 *   new MCPServer({
 *     name: "other",
 *     url: "https://other-server.com/mcp",
 *     headers: { "Authorization": "Bearer token" },
 *   }),
 * ]);
 *
 * const tools = await client.getTools();
 *
 * // Use with ChatDatabricks
 * const model = new ChatDatabricks({ endpoint: "..." });
 * const modelWithTools = model.bindTools(tools);
 * ```
 */
export class DatabricksMultiServerMCPClient {
  private readonly serverConfigs: Map<string, ServerInstance>;
  private underlyingClient: MultiServerMCPClient | null = null;
  private readonly options: Required<DatabricksMCPClientOptions>;
  private readonly servers: ServerInstance[];
  private initialized = false;

  constructor(servers: ServerInstance[], options: DatabricksMCPClientOptions = {}) {
    this.servers = servers;
    this.options = {
      throwOnLoadError: options.throwOnLoadError ?? true,
      prefixToolNameWithServerName: options.prefixToolNameWithServerName ?? false,
      additionalToolNamePrefix: options.additionalToolNamePrefix ?? "",
    };

    // Store server configs for later use (e.g., handleToolError)
    this.serverConfigs = new Map(servers.map((server) => [server.name, server]));
  }

  /**
   * Static factory method for creating the client with async initialization.
   * Preferred method for creating the client.
   */
  static async create(
    servers: ServerInstance[],
    options: DatabricksMCPClientOptions = {}
  ): Promise<DatabricksMultiServerMCPClient> {
    const client = new DatabricksMultiServerMCPClient(servers, options);
    await client.ensureInitialized();
    return client;
  }

  /**
   * Ensure the underlying MultiServerMCPClient is initialized.
   * This handles async initialization of DatabricksMCPServer connections.
   */
  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;

    // Build connections config for MultiServerMCPClient
    // Use headers-based auth for DatabricksMCPServer to avoid OAuth discovery issues
    const mcpServersConfig: Record<string, StreamableHTTPConnection> = {};
    for (const server of this.servers) {
      if (server instanceof DatabricksMCPServer) {
        // Use headers-based auth to bypass MCP SDK's OAuth flow
        mcpServersConfig[server.name] = await server.toConnectionConfigWithHeaders();
      } else {
        mcpServersConfig[server.name] = server.toConnectionConfig();
      }
    }

    // Create underlying MultiServerMCPClient
    this.underlyingClient = new MultiServerMCPClient({
      mcpServers: mcpServersConfig,
      throwOnLoadError: this.options.throwOnLoadError,
      prefixToolNameWithServerName: this.options.prefixToolNameWithServerName,
      additionalToolNamePrefix: this.options.additionalToolNamePrefix || undefined,
    });

    this.initialized = true;
  }

  /**
   * Get tools from MCP servers, applying handleToolError configuration.
   *
   * Loads tools from all servers in parallel for optimal performance.
   * Each tool's handleToolError is set based on the server's configuration.
   *
   * @param serverName - Optional server name to filter tools. If not provided, returns tools from all servers.
   * @returns Array of StructuredTool instances
   *
   * @example
   * ```typescript
   * // Get all tools
   * const allTools = await client.getTools();
   *
   * // Get tools from specific server
   * const databricksTools = await client.getTools("databricks");
   * ```
   */
  async getTools(serverName?: string): Promise<StructuredToolInterface[]> {
    await this.ensureInitialized();

    // Determine which servers to load from
    const serverNames = serverName ? [serverName] : Array.from(this.serverConfigs.keys());

    // Load tools from servers in parallel
    const toolLoadPromises = serverNames.map(async (name) => {
      const tools = await this.underlyingClient!.getTools(name);
      return { name, tools };
    });

    const results = await Promise.all(toolLoadPromises);

    // Collect tools from all servers
    const allTools: StructuredToolInterface[] = [];
    for (const { tools } of results) {
      allTools.push(...tools);
    }

    return allTools;
  }

  /**
   * Initialize connections to all configured servers.
   *
   * This method proactively establishes connections to all servers.
   * It's optional - connections will be established lazily on first use
   * if not called explicitly.
   *
   * @returns Record mapping server names to their loaded tools
   */
  async initializeConnections(): Promise<Record<string, StructuredToolInterface[]>> {
    await this.ensureInitialized();
    return this.underlyingClient!.initializeConnections();
  }

  /**
   * Get the underlying MCP client for a specific server.
   *
   * Useful for accessing prompts, resources, or other MCP features
   * not exposed through the simplified interface.
   *
   * @param serverName - Name of the server
   * @returns The MCP Client instance, or undefined if not found
   */
  async getClient(serverName: string): Promise<unknown> {
    await this.ensureInitialized();
    return this.underlyingClient!.getClient(serverName);
  }

  /**
   * Close all connections.
   *
   * Should be called when the client is no longer needed
   * to clean up resources.
   */
  async close(): Promise<void> {
    if (this.underlyingClient) {
      return this.underlyingClient.close();
    }
  }

  /**
   * Get the server configuration for a specific server.
   */
  getServerConfig(serverName: string): ServerInstance | undefined {
    return this.serverConfigs.get(serverName);
  }

  /**
   * Get all server names.
   */
  getServerNames(): string[] {
    return Array.from(this.serverConfigs.keys());
  }
}
