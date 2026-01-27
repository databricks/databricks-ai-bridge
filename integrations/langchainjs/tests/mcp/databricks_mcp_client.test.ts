/**
 * Unit tests for DatabricksMultiServerMCPClient
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { DatabricksMultiServerMCPClient } from "../../src/mcp/databricks_mcp_client.js";
import { MCPServer, DatabricksMCPServer } from "../../src/mcp/mcp_server.js";

// Mock @langchain/mcp-adapters
vi.mock("@langchain/mcp-adapters", () => ({
  MultiServerMCPClient: vi.fn().mockImplementation(() => ({
    getTools: vi.fn().mockResolvedValue([]),
    initializeConnections: vi.fn().mockResolvedValue({}),
    getClient: vi.fn().mockReturnValue(null),
    close: vi.fn().mockResolvedValue(undefined),
  })),
}));

describe("DatabricksMultiServerMCPClient", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("constructor", () => {
    it("creates client with single server", () => {
      const server = new MCPServer({
        name: "server1",
        url: "https://server1.com/mcp",
      });

      const client = new DatabricksMultiServerMCPClient([server]);

      expect(client.getServerNames()).toEqual(["server1"]);
    });

    it("creates client with multiple servers", () => {
      const servers = [
        new MCPServer({ name: "server1", url: "https://server1.com/mcp" }),
        new MCPServer({ name: "server2", url: "https://server2.com/mcp" }),
        new DatabricksMCPServer({ name: "databricks", url: "https://databricks.com/mcp" }),
      ];

      const client = new DatabricksMultiServerMCPClient(servers);

      expect(client.getServerNames()).toEqual(["server1", "server2", "databricks"]);
    });

    it("accepts options", () => {
      const server = new MCPServer({ name: "server1", url: "https://server1.com/mcp" });

      const client = new DatabricksMultiServerMCPClient([server], {
        throwOnLoadError: false,
        prefixToolNameWithServerName: true,
        additionalToolNamePrefix: "prefix_",
      });

      expect(client).toBeDefined();
    });
  });

  describe("getServerConfig", () => {
    it("returns server config by name", () => {
      const server = new MCPServer({
        name: "test-server",
        url: "https://test.com/mcp",
      });

      const client = new DatabricksMultiServerMCPClient([server]);
      const config = client.getServerConfig("test-server");

      expect(config).toBe(server);
    });

    it("returns undefined for non-existent server", () => {
      const server = new MCPServer({ name: "server1", url: "https://server1.com/mcp" });
      const client = new DatabricksMultiServerMCPClient([server]);

      expect(client.getServerConfig("non-existent")).toBeUndefined();
    });
  });

  describe("getTools", () => {
    it("returns tools from all servers", async () => {
      const { MultiServerMCPClient } = await import("@langchain/mcp-adapters");
      const mockGetTools = vi.fn().mockResolvedValue([
        { name: "tool1", description: "Tool 1" },
        { name: "tool2", description: "Tool 2" },
      ]);
      (MultiServerMCPClient as any).mockImplementation(() => ({
        getTools: mockGetTools,
        initializeConnections: vi.fn(),
        getClient: vi.fn(),
        close: vi.fn(),
      }));

      const servers = [
        new MCPServer({ name: "server1", url: "https://server1.com/mcp" }),
        new MCPServer({ name: "server2", url: "https://server2.com/mcp" }),
      ];

      const client = new DatabricksMultiServerMCPClient(servers);
      const tools = await client.getTools();

      expect(tools).toHaveLength(4); // 2 tools from each of 2 servers
      expect(mockGetTools).toHaveBeenCalledTimes(2);
    });

    it("returns tools from specific server", async () => {
      const { MultiServerMCPClient } = await import("@langchain/mcp-adapters");
      const mockGetTools = vi.fn().mockResolvedValue([{ name: "tool1" }]);
      (MultiServerMCPClient as any).mockImplementation(() => ({
        getTools: mockGetTools,
        initializeConnections: vi.fn(),
        getClient: vi.fn(),
        close: vi.fn(),
      }));

      const servers = [
        new MCPServer({ name: "server1", url: "https://server1.com/mcp" }),
        new MCPServer({ name: "server2", url: "https://server2.com/mcp" }),
      ];

      const client = new DatabricksMultiServerMCPClient(servers);
      const tools = await client.getTools("server1");

      expect(mockGetTools).toHaveBeenCalledWith("server1");
      expect(mockGetTools).toHaveBeenCalledTimes(1);
    });

  });

  describe("close", () => {
    it("closes underlying client after initialization", async () => {
      const { MultiServerMCPClient } = await import("@langchain/mcp-adapters");
      const mockClose = vi.fn().mockResolvedValue(undefined);
      (MultiServerMCPClient as any).mockImplementation(() => ({
        getTools: vi.fn().mockResolvedValue([]),
        initializeConnections: vi.fn(),
        getClient: vi.fn(),
        close: mockClose,
      }));

      const server = new MCPServer({ name: "server1", url: "https://server1.com/mcp" });
      const client = new DatabricksMultiServerMCPClient([server]);

      // Initialize the client first (lazy initialization)
      await client.getTools();
      await client.close();

      expect(mockClose).toHaveBeenCalled();
    });

    it("does not throw when closing uninitialized client", async () => {
      const server = new MCPServer({ name: "server1", url: "https://server1.com/mcp" });
      const client = new DatabricksMultiServerMCPClient([server]);

      // Should not throw even if client was never initialized
      await expect(client.close()).resolves.toBeUndefined();
    });
  });
});
