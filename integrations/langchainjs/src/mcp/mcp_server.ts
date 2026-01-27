/**
 * MCP Server configuration classes
 *
 * Provides MCPServer for generic MCP servers and DatabricksMCPServer
 * for servers requiring Databricks OAuth authentication.
 */

import { Config } from "@databricks/sdk-experimental";
import type { StreamableHTTPConnection } from "@langchain/mcp-adapters";
import type {
  MCPServerConfig,
  DatabricksMCPServerConfig,
  DatabricksMCPServerCreateConfig,
} from "./types.js";
import { DatabricksOAuthClientProvider } from "./databricks_oauth_provider.js";

/**
 * Base MCP server configuration for streamable HTTP transport.
 *
 * This class provides a typed interface for configuring MCP server connections.
 * All extra parameters are passed through to the underlying langchain-mcp-adapters
 * connection type, making this forward-compatible with future updates.
 *
 * @example
 * ```typescript
 * import { MCPServer, DatabricksMultiServerMCPClient } from "@databricks/langchainjs";
 *
 * const server = new MCPServer({
 *   name: "my-mcp-server",
 *   url: "https://my-mcp-server.com/mcp",
 *   headers: { "X-API-Key": "secret" },
 *   timeout: 30,
 *   handleToolError: true,
 * });
 *
 * const client = new DatabricksMultiServerMCPClient([server]);
 * const tools = await client.getTools();
 * ```
 */
export class MCPServer {
  readonly name: string;
  readonly url: string;
  readonly headers?: Record<string, string>;
  readonly timeout?: number;
  readonly sseReadTimeout?: number;

  // Store any additional properties for forward compatibility
  protected readonly extraConfig: Record<string, unknown>;

  constructor(config: MCPServerConfig & Record<string, unknown>) {
    const { name, url, headers, timeout, sseReadTimeout, ...extra } = config;

    this.name = name;
    this.url = url;
    this.headers = headers;
    this.timeout = timeout;
    this.sseReadTimeout = sseReadTimeout;
    this.extraConfig = extra;
  }

  /**
   * Convert to connection dictionary for MultiServerMCPClient.
   * Returns the format expected by @langchain/mcp-adapters.
   */
  toConnectionConfig(): StreamableHTTPConnection {
    const config: StreamableHTTPConnection = {
      transport: "http",
      url: this.url,
      ...this.extraConfig,
    };

    if (this.headers) {
      config.headers = this.headers;
    }
    if (this.timeout !== undefined) {
      config.defaultToolTimeout = this.timeout * 1000; // Convert seconds to milliseconds
    }
    // Note: sseReadTimeout is kept in extraConfig if passed, for forward compatibility

    return config;
  }
}

/**
 * MCP server with Databricks OAuth authentication.
 *
 * Automatically configures OAuth authentication using the Databricks SDK.
 * This handles all Databricks authentication methods (PAT, OAuth, Azure MSI, etc.)
 * transparently through the SDK's authentication chain.
 *
 * @example
 * ```typescript
 * import { DatabricksMCPServer, DatabricksMultiServerMCPClient } from "@databricks/langchainjs";
 *
 * // Using default SDK authentication (env vars, CLI, etc.)
 * const server = new DatabricksMCPServer({
 *   name: "databricks-mcp",
 *   url: "https://my-workspace.databricks.com/api/mcp",
 * });
 *
 * // With M2M OAuth (service principal)
 * const serverWithM2M = new DatabricksMCPServer({
 *   name: "databricks-mcp",
 *   url: "https://my-workspace.databricks.com/api/mcp",
 *   auth: {
 *     authType: "oauth-m2m",
 *     host: "https://my-workspace.databricks.com",
 *     clientId: "your-client-id",
 *     clientSecret: "your-client-secret",
 *   },
 *   handleToolError: true,
 * });
 *
 * // With personal access token
 * const serverWithPAT = new DatabricksMCPServer({
 *   name: "databricks-mcp",
 *   url: "https://my-workspace.databricks.com/api/mcp",
 *   auth: {
 *     host: "https://my-workspace.databricks.com",
 *     token: "dapi...",
 *   },
 * });
 *
 * const client = new DatabricksMultiServerMCPClient([server]);
 * const tools = await client.getTools();
 * ```
 */
export class DatabricksMCPServer extends MCPServer {
  private readonly auth: Config;
  private readonly authProvider: DatabricksOAuthClientProvider;

  constructor(config: DatabricksMCPServerConfig & Record<string, unknown>) {
    // Extract Databricks-specific config before passing to base
    const { auth, ...baseConfig } = config;
    super(baseConfig);

    // Initialize Databricks SDK config from options
    this.auth = new Config(auth ?? {});

    // Create OAuth provider for MCP authentication
    this.authProvider = new DatabricksOAuthClientProvider(this.auth);
  }

  /**
   * Initialize authentication and get the Bearer token.
   * Call this before using toConnectionConfig() to ensure auth is ready.
   */
  async initializeAuth(): Promise<string> {
    await this.auth.ensureResolved();

    const headers = new Headers();
    await this.auth.authenticate(headers);
    const authHeader = headers.get("Authorization");

    if (!authHeader?.startsWith("Bearer ")) {
      throw new Error("Invalid authentication token format. Expected Bearer token.");
    }
    return authHeader;
  }

  /**
   * Convert to connection dictionary, including Databricks OAuth provider.
   */
  override toConnectionConfig(): StreamableHTTPConnection {
    const baseConfig = super.toConnectionConfig();

    return {
      ...baseConfig,
      authProvider: this.authProvider,
    };
  }

  /**
   * Convert to connection dictionary using headers-based authentication.
   * This bypasses the MCP SDK's OAuth flow and uses the Bearer token directly.
   * Use this when connecting to servers that don't support OAuth discovery.
   */
  async toConnectionConfigWithHeaders(): Promise<StreamableHTTPConnection> {
    const baseConfig = super.toConnectionConfig();
    const authHeader = await this.initializeAuth();

    return {
      ...baseConfig,
      headers: {
        ...baseConfig.headers,
        Authorization: authHeader,
      },
      // Don't use authProvider - use headers directly
    };
  }

  /**
   * Factory method to create a DatabricksMCPServer with a path.
   * The host is resolved from auth (or environment variables).
   *
   * @param config - Server configuration with name, path, and optional auth
   * @returns DatabricksMCPServer configured with the resolved host
   *
   * @example
   * ```typescript
   * // Using default auth - host from DATABRICKS_HOST env var
   * const server = await DatabricksMCPServer.create({
   *   name: "sql",
   *   path: "/api/2.0/mcp/sql",
   * });
   *
   * // Using explicit config
   * const server = await DatabricksMCPServer.create({
   *   name: "sql",
   *   path: "/api/2.0/mcp/sql",
   *   auth: {
   *     host: "https://my-workspace.databricks.com",
   *     token: "dapi...",
   *   },
   * });
   * ```
   */
  static async create(
    config: DatabricksMCPServerCreateConfig & Record<string, unknown>
  ): Promise<DatabricksMCPServer> {
    const { name, path, auth, ...rest } = config;
    const sdkConfig = new Config(auth ?? {});

    const host = (await sdkConfig.getHost())?.toString().replace(/\/$/, "");
    if (!host) {
      throw new Error(
        "Databricks host not configured. Set DATABRICKS_HOST environment variable or provide host in auth."
      );
    }

    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    const url = `${host}${normalizedPath}`;

    return new DatabricksMCPServer({
      name,
      url,
      auth,
      ...rest,
    });
  }

  /**
   * Factory method to create a DatabricksMCPServer for a Unity Catalog function.
   *
   * @param catalog - UC catalog name
   * @param schema - UC schema name
   * @param functionName - UC function name (optional, for specific function)
   * @param options - Additional server configuration options
   * @returns DatabricksMCPServer configured for the UC function
   *
   * @example
   * ```typescript
   * const server = await DatabricksMCPServer.fromUCFunction(
   *   "my_catalog",
   *   "my_schema",
   *   "my_function",
   *   { handleToolError: true }
   * );
   * ```
   */
  static async fromUCFunction(
    catalog: string,
    schema: string,
    functionName?: string,
    options: Partial<Omit<DatabricksMCPServerConfig, "name" | "url">> = {}
  ): Promise<DatabricksMCPServer> {
    const sdkConfig = new Config(options.auth ?? {});

    // Build the UC function MCP endpoint URL
    const host = (await sdkConfig.getHost())?.toString().replace(/\/$/, "");
    if (!host) {
      throw new Error(
        "Databricks host not configured. Set DATABRICKS_HOST environment variable or provide host in auth."
      );
    }
    const functionPath = functionName
      ? `${catalog}/${schema}/${functionName}`
      : `${catalog}/${schema}`;
    const url = `${host}/api/2.0/mcp/functions/${functionPath}`;

    const name = functionName
      ? `uc-function-${catalog}-${schema}-${functionName}`
      : `uc-functions-${catalog}-${schema}`;

    return new DatabricksMCPServer({
      name,
      url,
      ...options,
    });
  }

  /**
   * Factory method to create a DatabricksMCPServer for a Vector Search index.
   *
   * @param catalog - UC catalog name
   * @param schema - UC schema name
   * @param indexName - Vector Search index name (optional, for specific index)
   * @param options - Additional server configuration options
   * @returns DatabricksMCPServer configured for the Vector Search index
   *
   * @example
   * ```typescript
   * const server = DatabricksMCPServer.fromVectorSearch(
   *   "my_catalog",
   *   "my_schema",
   *   "my_index",
   *   { handleToolError: true }
   * );
   * ```
   */
  static async fromVectorSearch(
    catalog: string,
    schema: string,
    indexName?: string,
    options: Partial<Omit<DatabricksMCPServerConfig, "name" | "url">> = {}
  ): Promise<DatabricksMCPServer> {
    const sdkConfig = new Config(options.auth ?? {});

    // Build the Vector Search MCP endpoint URL
    const host = (await sdkConfig.getHost())?.toString().replace(/\/$/, "");
    if (!host) {
      throw new Error(
        "Databricks host not configured. Set DATABRICKS_HOST environment variable or provide host in auth."
      );
    }
    const url = indexName
      ? `${host}/api/2.0/mcp/vector-search/${catalog}/${schema}/${indexName}`
      : `${host}/api/2.0/mcp/vector-search/${catalog}/${schema}`;

    const name = indexName
      ? `vector-search-${catalog}-${schema}-${indexName}`
      : `vector-search-${catalog}-${schema}`;

    return new DatabricksMCPServer({
      name,
      url,
      ...options,
    });
  }

  /**
   * Factory method to create a DatabricksMCPServer for a Genie Space.
   *
   * @param spaceId - Genie Space ID
   * @param options - Additional server configuration options
   * @returns DatabricksMCPServer configured for the Genie Space
   *
   * @example
   * ```typescript
   * const server = await DatabricksMCPServer.fromGenieSpace(
   *   "01ef19c578b21dc6af6e10983fb1e3f9",
   *   { handleToolError: true }
   * );
   * ```
   */
  static async fromGenieSpace(
    spaceId: string,
    options: Partial<Omit<DatabricksMCPServerConfig, "name" | "url">> = {}
  ): Promise<DatabricksMCPServer> {
    const sdkConfig = new Config(options.auth ?? {});

    // Build the Genie Space MCP endpoint URL
    const host = (await sdkConfig.getHost())?.toString().replace(/\/$/, "");
    if (!host) {
      throw new Error(
        "Databricks host not configured. Set DATABRICKS_HOST environment variable or provide host in auth."
      );
    }
    const url = `${host}/api/2.0/mcp/genie/${spaceId}`;

    const name = `genie-space-${spaceId}`;

    return new DatabricksMCPServer({
      name,
      url,
      ...options,
    });
  }
}
