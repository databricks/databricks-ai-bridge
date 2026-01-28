/**
 * Unit tests for buildMCPServerConfig
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

// Use vi.hoisted to ensure the mock is properly hoisted
const { mockAuthenticate } = vi.hoisted(() => {
  const mockAuthenticate = vi.fn((headers: Headers) => {
    headers.set("Authorization", "Bearer test-token");
    return Promise.resolve();
  });
  return { mockAuthenticate };
});

vi.mock("@databricks/sdk-experimental", () => ({
  Config: vi.fn(() => ({
    ensureResolved: vi.fn().mockResolvedValue(undefined),
    getHost: vi.fn().mockResolvedValue("https://test-workspace.databricks.com"),
    authenticate: mockAuthenticate,
  })),
}));

import { buildMCPServerConfig } from "../../src/mcp/databricks_mcp_client.js";
import { MCPServer, DatabricksMCPServer } from "../../src/mcp/mcp_server.js";

describe("buildMCPServerConfig", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("builds config from MCPServer instances", async () => {
    const servers = [
      new MCPServer({ name: "server1", url: "https://server1.com/mcp" }),
      new MCPServer({ name: "server2", url: "https://server2.com/mcp" }),
    ];

    const config = await buildMCPServerConfig(servers);

    expect(Object.keys(config)).toEqual(["server1", "server2"]);
    expect(config.server1.url).toBe("https://server1.com/mcp");
    expect(config.server1.transport).toBe("http");
    expect(config.server2.url).toBe("https://server2.com/mcp");
  });

  it("builds config from DatabricksMCPServer instances with resolved URL and authProvider", async () => {
    const server = new DatabricksMCPServer({
      name: "databricks",
      path: "/api/2.0/mcp/sql",
    });

    const config = await buildMCPServerConfig([server]);

    expect(Object.keys(config)).toEqual(["databricks"]);
    expect(config.databricks.url).toBe("https://test-workspace.databricks.com/api/2.0/mcp/sql");
    expect(config.databricks.transport).toBe("http");
    // Auth is provided via authProvider, not headers
    expect(config.databricks.authProvider).toBeDefined();
  });

  it("builds config from mixed server types", async () => {
    const servers = [
      new MCPServer({ name: "external", url: "https://external.com/mcp" }),
      new DatabricksMCPServer({ name: "databricks", path: "/api/2.0/mcp/sql" }),
    ];

    const config = await buildMCPServerConfig(servers);

    expect(Object.keys(config)).toEqual(["external", "databricks"]);
    expect(config.external.url).toBe("https://external.com/mcp");
    expect(config.external.headers).toBeUndefined();
    expect(config.external.authProvider).toBeUndefined();
    expect(config.databricks.url).toBe("https://test-workspace.databricks.com/api/2.0/mcp/sql");
    // DatabricksMCPServer uses authProvider instead of headers
    expect(config.databricks.authProvider).toBeDefined();
  });

  it("includes headers from MCPServer config", async () => {
    const server = new MCPServer({
      name: "server",
      url: "https://server.com/mcp",
      headers: { "X-API-Key": "secret" },
    });

    const config = await buildMCPServerConfig([server]);

    expect(config.server.headers).toEqual({ "X-API-Key": "secret" });
  });

  it("converts timeout to milliseconds", async () => {
    const server = new MCPServer({
      name: "server",
      url: "https://server.com/mcp",
      timeout: 30,
    });

    const config = await buildMCPServerConfig([server]);

    expect(config.server.defaultToolTimeout).toBe(30000);
  });

  it("returns empty config for empty server array", async () => {
    const config = await buildMCPServerConfig([]);

    expect(config).toEqual({});
  });
});
