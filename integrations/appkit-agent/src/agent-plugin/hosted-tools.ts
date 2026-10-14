/**
 * OpenResponses-style hosted tool definitions for Databricks services.
 *
 * These types follow the OpenResponses convention of discriminating on `type`.
 * Internally, each hosted tool is resolved to a DatabricksMCPServer instance
 * so the agent can call managed MCP endpoints on the workspace.
 */

// ---------------------------------------------------------------------------
// Hosted tool type definitions
// ---------------------------------------------------------------------------

export interface GenieTool {
  type: "genie-space";
  genie_space: { id: string };
}

export interface VectorSearchIndexTool {
  type: "vector_search_index";
  vector_search_index: { name: string };
}

export interface CustomMcpServerTool {
  type: "custom_mcp_server";
  custom_mcp_server: { app_name: string; app_url: string };
}

export interface ExternalMcpServerTool {
  type: "external_mcp_server";
  external_mcp_server: { connection_name: string };
}

export type HostedTool =
  | GenieTool
  | VectorSearchIndexTool
  | CustomMcpServerTool
  | ExternalMcpServerTool;

// ---------------------------------------------------------------------------
// Type guard
// ---------------------------------------------------------------------------

const HOSTED_TOOL_TYPES = new Set([
  "genie-space",
  "vector_search_index",
  "custom_mcp_server",
  "external_mcp_server",
]);

export function isHostedTool(t: unknown): t is HostedTool {
  return (
    typeof t === "object" &&
    t !== null &&
    HOSTED_TOOL_TYPES.has((t as any).type)
  );
}

// ---------------------------------------------------------------------------
// Resolver: HostedTool → DatabricksMCPServer
// ---------------------------------------------------------------------------

/**
 * Resolve an array of HostedTool definitions into DatabricksMCPServer instances.
 *
 * Uses factory methods from `@databricks/langchainjs` where available, and
 * falls back to raw DatabricksMCPServer construction for tool types that
 * map to known managed MCP API paths.
 */
export async function resolveHostedTools(
  tools: HostedTool[],
): Promise<InstanceType<any>[]> {
  const { DatabricksMCPServer } = await import("@databricks/langchainjs");

  return tools.map((tool) => {
    switch (tool.type) {
      case "genie-space":
        return DatabricksMCPServer.fromGenieSpace(tool.genie_space.id);

      case "vector_search_index": {
        const parts = tool.vector_search_index.name.split(".");
        if (parts.length !== 3) {
          throw new Error(
            `vector_search_index name must be "catalog.schema.index", got "${tool.vector_search_index.name}"`,
          );
        }
        const [catalog, schema, index] = parts;
        return DatabricksMCPServer.fromVectorSearch(catalog, schema, index);
      }

      case "custom_mcp_server":
        return new DatabricksMCPServer({
          name: `mcp-app-${tool.custom_mcp_server.app_name}`,
          path: `/apps/${tool.custom_mcp_server.app_url}`,
        });

      case "external_mcp_server":
        return new DatabricksMCPServer({
          name: `mcp-ext-${tool.external_mcp_server.connection_name}`,
          path: `/api/2.0/mcp/connections/${tool.external_mcp_server.connection_name}`,
        });

      default: {
        throw new Error(`Unknown hosted tool type: ${(tool as any).type}`);
      }
    }
  });
}
