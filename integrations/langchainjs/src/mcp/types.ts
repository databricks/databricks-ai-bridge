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
 * Base configuration options shared by all MCP server types.
 */
export interface BaseMCPServerConfig {
  /** Unique name to identify this server connection */
  name: string;

  /** Custom HTTP headers to include in requests */
  headers?: Record<string, string>;

  /** Request timeout in seconds */
  timeout?: number;

  /** SSE read timeout in seconds */
  sseReadTimeout?: number;
}

/**
 * Configuration for generic MCP servers.
 * Uses streamable HTTP transport.
 */
export interface MCPServerConfig extends BaseMCPServerConfig {
  /** MCP server URL endpoint */
  url: string;
}

/**
 * Configuration for Databricks MCP servers.
 * Uses path instead of full URL - host is resolved from auth config.
 */
export interface DatabricksMCPServerConfig extends BaseMCPServerConfig {
  /** API path (e.g., "/api/2.0/mcp/sql"). Host is resolved from auth. */
  path: string;

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
  auth?: DatabricksConfigOptions;
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
