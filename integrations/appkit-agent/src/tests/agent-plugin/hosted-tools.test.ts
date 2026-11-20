import { beforeEach, describe, expect, test, vi } from "vitest";
import type {
  CustomMcpServerTool,
  ExternalMcpServerTool,
  GenieTool,
  HostedTool,
  VectorSearchIndexTool,
} from "../../agent-plugin/hosted-tools";
import {
  isHostedTool,
  resolveHostedTools,
} from "../../agent-plugin/hosted-tools";

const genieTool: GenieTool = {
  type: "genie-space",
  genie_space: { id: "space-123" },
};

const vectorSearchTool: VectorSearchIndexTool = {
  type: "vector_search_index",
  vector_search_index: { name: "catalog.schema.my_index" },
};

const customMcpTool: CustomMcpServerTool = {
  type: "custom_mcp_server",
  custom_mcp_server: { app_name: "my-mcp-app", app_url: "my-mcp-app" },
};

const externalMcpTool: ExternalMcpServerTool = {
  type: "external_mcp_server",
  external_mcp_server: { connection_name: "my-connection" },
};

// ---------------------------------------------------------------------------
// isHostedTool
// ---------------------------------------------------------------------------

describe("isHostedTool", () => {
  test("returns true for GenieTool", () => {
    expect(isHostedTool(genieTool)).toBe(true);
  });

  test("returns true for VectorSearchIndexTool", () => {
    expect(isHostedTool(vectorSearchTool)).toBe(true);
  });

  test("returns true for CustomMcpServerTool", () => {
    expect(isHostedTool(customMcpTool)).toBe(true);
  });

  test("returns true for ExternalMcpServerTool", () => {
    expect(isHostedTool(externalMcpTool)).toBe(true);
  });

  test("returns false for FunctionTool", () => {
    const functionTool = {
      type: "function",
      name: "test",
      execute: async () => "result",
    };
    expect(isHostedTool(functionTool)).toBe(false);
  });

  test("returns false for null/undefined", () => {
    expect(isHostedTool(null)).toBe(false);
    expect(isHostedTool(undefined)).toBe(false);
  });

  test("returns false for object with unknown type", () => {
    expect(isHostedTool({ type: "unknown_tool" })).toBe(false);
  });

  test("returns false for non-object", () => {
    expect(isHostedTool("genie")).toBe(false);
    expect(isHostedTool(42)).toBe(false);
  });
});

describe("hosted tool types", () => {
  test("all hosted tools satisfy HostedTool union", () => {
    const tools: HostedTool[] = [
      genieTool,
      vectorSearchTool,
      customMcpTool,
      externalMcpTool,
    ];

    expect(tools).toHaveLength(4);
    for (const tool of tools) {
      expect(isHostedTool(tool)).toBe(true);
    }
  });

  test("can be mixed in an array with discriminator", () => {
    const tools: HostedTool[] = [genieTool, vectorSearchTool];
    const types = tools.map((t) => t.type);
    expect(types).toEqual(["genie-space", "vector_search_index"]);
  });
});

// ---------------------------------------------------------------------------
// resolveHostedTools
// ---------------------------------------------------------------------------

const { mockFromGenieSpace, mockFromVectorSearch, MockDatabricksMCPServer } =
  vi.hoisted(() => {
    const mockFromGenieSpace = vi.fn().mockReturnValue({ _type: "genie" });
    const mockFromVectorSearch = vi.fn().mockReturnValue({ _type: "vector" });
    const constructorSpy = vi.fn();
    class MockDatabricksMCPServer {
      name: string;
      path: string;
      constructor(opts: any) {
        constructorSpy(opts);
        this.name = opts.name;
        this.path = opts.path;
      }
      static fromGenieSpace = mockFromGenieSpace;
      static fromVectorSearch = mockFromVectorSearch;
    }
    return {
      mockFromGenieSpace,
      mockFromVectorSearch,
      MockDatabricksMCPServer: Object.assign(MockDatabricksMCPServer, {
        _constructorSpy: constructorSpy,
      }),
    };
  });

vi.mock("@databricks/langchainjs", () => ({
  DatabricksMCPServer: MockDatabricksMCPServer,
}));

describe("resolveHostedTools", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("resolves genie-space tool via fromGenieSpace", async () => {
    const result = await resolveHostedTools([genieTool]);

    expect(mockFromGenieSpace).toHaveBeenCalledWith("space-123");
    expect(result).toHaveLength(1);
    expect(result[0]._type).toBe("genie");
  });

  test("resolves vector_search_index via fromVectorSearch", async () => {
    const result = await resolveHostedTools([vectorSearchTool]);

    expect(mockFromVectorSearch).toHaveBeenCalledWith(
      "catalog",
      "schema",
      "my_index",
    );
    expect(result).toHaveLength(1);
    expect(result[0]._type).toBe("vector");
  });

  test("throws for vector_search_index with wrong part count", async () => {
    const badTool: VectorSearchIndexTool = {
      type: "vector_search_index",
      vector_search_index: { name: "just.two" },
    };

    await expect(resolveHostedTools([badTool])).rejects.toThrow(
      'vector_search_index name must be "catalog.schema.index"',
    );
  });

  test("resolves custom_mcp_server via constructor", async () => {
    const result = await resolveHostedTools([customMcpTool]);

    expect(MockDatabricksMCPServer._constructorSpy).toHaveBeenCalledWith({
      name: "mcp-app-my-mcp-app",
      path: "/apps/my-mcp-app",
    });
    expect(result).toHaveLength(1);
  });

  test("resolves external_mcp_server via constructor", async () => {
    const result = await resolveHostedTools([externalMcpTool]);

    expect(MockDatabricksMCPServer._constructorSpy).toHaveBeenCalledWith({
      name: "mcp-ext-my-connection",
      path: "/api/2.0/mcp/connections/my-connection",
    });
    expect(result).toHaveLength(1);
  });

  test("resolves multiple tools in order", async () => {
    const result = await resolveHostedTools([genieTool, customMcpTool]);

    expect(result).toHaveLength(2);
    expect(mockFromGenieSpace).toHaveBeenCalled();
    expect(MockDatabricksMCPServer._constructorSpy).toHaveBeenCalled();
  });
});
