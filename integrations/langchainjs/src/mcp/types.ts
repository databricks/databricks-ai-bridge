/**
 * Type definitions for MCP (Model Context Protocol) integration
 */

import type { Config } from "@databricks/sdk-experimental";

type InferConstructorParams<T> = T extends new (arg: infer P) => any ? P : never;

/**
 * Databricks SDK configuration options.
 * This mirrors the ConfigOptions type from @databricks/sdk-experimental
 * so users don't need to install the SDK directly.
 */
export type DatabricksConfigOptions = InferConstructorParams<typeof Config>;

/**
 * Base server configuration options for MCP servers.
 * Uses streamable HTTP transport (aligned with Python implementation).
 */
export interface MCPServerConfig {
  /** Unique name to identify this server connection */
  name: string;

  /** MCP server URL endpoint */
  url: string;

  /** Custom HTTP headers to include in requests */
  headers?: Record<string, string>;

  /** Request timeout in seconds */
  timeout?: number;

  /** SSE read timeout in seconds */
  sseReadTimeout?: number;
}

/**
 * Databricks-specific MCP server configuration.
 * Extends base config with Databricks SDK authentication.
 */
export interface DatabricksMCPServerConfig extends MCPServerConfig {
  /**
   * Databricks SDK configuration options for authentication.
   * If not provided, will use default SDK authentication chain
   * (environment variables, CLI config, etc.).
   *
   * @example
   * ```typescript
   * // Use M2M OAuth with service principal
   * {
   *   authType: "oauth-m2m",
   *   clientId: "your-client-id",
   *   clientSecret: "your-client-secret",
   *   host: "https://your-workspace.databricks.com"
   * }
   *
   * // Use personal access token
   * {
   *   host: "https://your-workspace.databricks.com",
   *   token: "dapi..."
   * }
   *
   * // Use default auth chain (reads from env vars, CLI config, etc.)
   * {} // or omit entirely
   * ```
   */
  databricksConfig?: DatabricksConfigOptions;
}

/**
 * Options for DatabricksMultiServerMCPClient
 */
export interface DatabricksMCPClientOptions {
  /**
   * Throw error if any tool fails to load.
   * @default true
   */
  throwOnLoadError?: boolean;

  /**
   * Prefix tool names with server name to avoid conflicts.
   * @default false
   */
  prefixToolNameWithServerName?: boolean;

  /**
   * Additional prefix for all tool names.
   */
  additionalToolNamePrefix?: string;
}

/**
 * Type guard to check if server config is DatabricksMCPServerConfig
 */
export function isDatabricksMCPServerConfig(
  config: MCPServerConfig | DatabricksMCPServerConfig
): config is DatabricksMCPServerConfig {
  return "config" in config;
}
