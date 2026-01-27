/**
 * MCP Server configuration classes
 *
 * Provides MCPServer for generic MCP servers and DatabricksMCPServer
 * for servers requiring Databricks OAuth authentication.
 */

import { Config } from "@databricks/sdk-experimental";
import type { StreamableHTTPConnection } from "@langchain/mcp-adapters";
import type {
  BaseMCPServerConfig,
  MCPServerConfig,
  DatabricksMCPServerConfig,
  DatabricksConfigOptions,
} from "./types.js";
import { DatabricksOAuthClientProvider } from "./databricks_oauth_provider.js";

/**
 * Abstract base class for MCP server configurations.
 *
 * Provides common properties and the interface for connection config generation.
 * Subclasses implement `toConnectionConfig()` to provide transport-specific configuration.
 */
abstract class BaseMCPServer {
  readonly name: string;
  readonly headers?: Record<string, string>;
  readonly timeout?: number;
  readonly sseReadTimeout?: number;

  constructor(config: BaseMCPServerConfig) {
    this.name = config.name;
    this.headers = config.headers;
    this.timeout = config.timeout;
    this.sseReadTimeout = config.sseReadTimeout;
  }

  /**
   * Convert to connection dictionary for MultiServerMCPClient.
   * Returns the format expected by @langchain/mcp-adapters.
   */
  abstract toConnectionConfig(): StreamableHTTPConnection;

  /**
   * Build base connection config with common properties.
   */
  protected buildBaseConfig(url: string): StreamableHTTPConnection {
    const config: StreamableHTTPConnection = {
      transport: "http",
      url,
    };

    if (this.headers) {
      config.headers = this.headers;
    }
    if (this.timeout !== undefined) {
      config.defaultToolTimeout = this.timeout * 1000; // Convert seconds to milliseconds
    }

    return config;
  }
}

/**
 * MCP server configuration for streamable HTTP transport.
 *
 * Use this for generic (non-Databricks) MCP servers where you have a full URL.
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
 * });
 *
 * const client = new DatabricksMultiServerMCPClient([server]);
 * const tools = await client.getTools();
 * ```
 */
export class MCPServer extends BaseMCPServer {
  readonly url: string;

  constructor(config: MCPServerConfig) {
    super(config);
    this.url = config.url;
  }

  override toConnectionConfig(): StreamableHTTPConnection {
    return this.buildBaseConfig(this.url);
  }
}

/**
 * MCP server with Databricks OAuth authentication.
 *
 * Automatically configures OAuth authentication using the Databricks SDK.
 * This handles all Databricks authentication methods (PAT, OAuth, Azure MSI, etc.)
 * transparently through the SDK's authentication chain.
 *
 * The host is resolved lazily from the Databricks SDK config (via DATABRICKS_HOST
 * environment variable, CLI config, or explicit auth config).
 *
 * @example
 * ```typescript
 * import { DatabricksMCPServer, DatabricksMultiServerMCPClient } from "@databricks/langchainjs";
 *
 * // Host resolved from DATABRICKS_HOST env var or auth config
 * const server = new DatabricksMCPServer({
 *   name: "databricks-mcp",
 *   path: "/api/2.0/mcp/sql",
 * });
 *
 * // With explicit auth (host resolved from auth config)
 * const serverWithAuth = new DatabricksMCPServer({
 *   name: "databricks-mcp",
 *   path: "/api/2.0/mcp/sql",
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
export class DatabricksMCPServer extends BaseMCPServer {
  private readonly sdkConfig: Config;
  private readonly authProvider: DatabricksOAuthClientProvider;
  private readonly path: string;
  private resolvedUrl?: string;

  constructor(config: DatabricksMCPServerConfig) {
    super(config);

    // Normalize and store path
    this.path = config.path.startsWith("/") ? config.path : `/${config.path}`;

    // Initialize Databricks SDK config from options
    this.sdkConfig = new Config(config.auth ?? {});

    // Create OAuth provider for MCP authentication
    this.authProvider = new DatabricksOAuthClientProvider(this.sdkConfig);
  }

  /**
   * Resolve the full URL by combining the host from SDK config with the path.
   */
  private async resolveUrl(): Promise<string> {
    if (this.resolvedUrl) {
      return this.resolvedUrl;
    }

    const host = (await this.sdkConfig.getHost())?.toString().replace(/\/$/, "");
    if (!host) {
      throw new Error(
        "Databricks host not configured. Set DATABRICKS_HOST environment variable or provide host in auth."
      );
    }

    this.resolvedUrl = `${host}${this.path}`;
    return this.resolvedUrl;
  }

  /**
   * Initialize authentication and get the Bearer token.
   */
  async initializeAuth(): Promise<string> {
    await this.sdkConfig.ensureResolved();

    const headers = new Headers();
    await this.sdkConfig.authenticate(headers);
    const authHeader = headers.get("Authorization");

    if (!authHeader?.startsWith("Bearer ")) {
      throw new Error("Invalid authentication token format. Expected Bearer token.");
    }
    return authHeader;
  }

  /**
   * Convert to connection dictionary, including Databricks OAuth provider.
   * Note: URL is empty here as it's resolved lazily in toConnectionConfigWithHeaders().
   */
  override toConnectionConfig(): StreamableHTTPConnection {
    return {
      ...this.buildBaseConfig(""),
      authProvider: this.authProvider,
    };
  }

  /**
   * Convert to connection dictionary using headers-based authentication.
   * This resolves the URL and adds the Bearer token directly to headers.
   * Use this when connecting to servers that don't support OAuth discovery.
   */
  async toConnectionConfigWithHeaders(): Promise<StreamableHTTPConnection> {
    const resolvedUrl = await this.resolveUrl();
    const authHeader = await this.initializeAuth();

    const config = this.buildBaseConfig(resolvedUrl);
    config.headers = {
      ...config.headers,
      Authorization: authHeader,
    };

    return config;
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
   * const server = DatabricksMCPServer.fromUCFunction(
   *   "my_catalog",
   *   "my_schema",
   *   "my_function"
   * );
   * ```
   */
  static fromUCFunction(
    catalog: string,
    schema: string,
    functionName?: string,
    options: { auth?: DatabricksConfigOptions; timeout?: number } = {}
  ): DatabricksMCPServer {
    const functionPath = functionName
      ? `${catalog}/${schema}/${functionName}`
      : `${catalog}/${schema}`;

    const name = functionName
      ? `uc-function-${catalog}-${schema}-${functionName}`
      : `uc-functions-${catalog}-${schema}`;

    return new DatabricksMCPServer({
      name,
      path: `/api/2.0/mcp/functions/${functionPath}`,
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
   *   "my_index"
   * );
   * ```
   */
  static fromVectorSearch(
    catalog: string,
    schema: string,
    indexName?: string,
    options: { auth?: DatabricksConfigOptions; timeout?: number } = {}
  ): DatabricksMCPServer {
    const indexPath = indexName
      ? `${catalog}/${schema}/${indexName}`
      : `${catalog}/${schema}`;

    const name = indexName
      ? `vector-search-${catalog}-${schema}-${indexName}`
      : `vector-search-${catalog}-${schema}`;

    return new DatabricksMCPServer({
      name,
      path: `/api/2.0/mcp/vector-search/${indexPath}`,
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
   * const server = DatabricksMCPServer.fromGenieSpace(
   *   "01ef19c578b21dc6af6e10983fb1e3f9"
   * );
   * ```
   */
  static fromGenieSpace(
    spaceId: string,
    options: { auth?: DatabricksConfigOptions; timeout?: number } = {}
  ): DatabricksMCPServer {
    const name = `genie-space-${spaceId}`;

    return new DatabricksMCPServer({
      name,
      path: `/api/2.0/mcp/genie/${spaceId}`,
      ...options,
    });
  }
}
