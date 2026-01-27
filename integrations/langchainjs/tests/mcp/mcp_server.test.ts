/**
 * Unit tests for MCPServer and DatabricksMCPServer
 */

import { describe, it, expect } from "vitest";
import { MCPServer, DatabricksMCPServer } from "../../src/mcp/mcp_server.js";

describe("MCPServer", () => {
  it("creates server with minimal config", () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
    });

    expect(server.name).toBe("test-server");
    expect(server.url).toBe("https://example.com/mcp");
    expect(server.headers).toBeUndefined();
    expect(server.timeout).toBeUndefined();
  });

  it("creates server with full config", () => {
    const server = new MCPServer({
      name: "full-server",
      url: "https://example.com/mcp",
      headers: { "X-API-Key": "secret" },
      timeout: 30,
      sseReadTimeout: 60,
    });

    expect(server.name).toBe("full-server");
    expect(server.url).toBe("https://example.com/mcp");
    expect(server.headers).toEqual({ "X-API-Key": "secret" });
    expect(server.timeout).toBe(30);
    expect(server.sseReadTimeout).toBe(60);
  });

  it("converts to connection config with http transport", () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
    });

    const config = server.toConnectionConfig();

    expect(config.transport).toBe("http");
    expect(config.url).toBe("https://example.com/mcp");
  });

  it("converts timeout to milliseconds in connection config", () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
      timeout: 30,
    });

    const config = server.toConnectionConfig();

    expect(config.defaultToolTimeout).toBe(30000); // 30 seconds in ms
  });

  it("includes headers in connection config", () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
      headers: { Authorization: "Bearer token123" },
    });

    const config = server.toConnectionConfig();

    expect(config.headers).toEqual({ Authorization: "Bearer token123" });
  });

  it("excludes name from connection config", () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
    });

    const config = server.toConnectionConfig();

    expect(config).not.toHaveProperty("name");
  });

  it("preserves extra config properties", () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
      customProperty: "custom-value",
    } as any);

    const config = server.toConnectionConfig();

    expect(config.customProperty).toBe("custom-value");
  });
});

describe("DatabricksMCPServer", () => {
  it("creates server with minimal config", () => {
    const server = new DatabricksMCPServer({
      name: "databricks-server",
      url: "https://workspace.databricks.com/api/mcp",
    });

    expect(server.name).toBe("databricks-server");
    expect(server.url).toBe("https://workspace.databricks.com/api/mcp");
  });

  it("includes authProvider in connection config", () => {
    const server = new DatabricksMCPServer({
      name: "databricks-server",
      url: "https://workspace.databricks.com/api/mcp",
    });

    const config = server.toConnectionConfig();

    expect(config).toHaveProperty("authProvider");
    expect(config.transport).toBe("http");
    expect(config.url).toBe("https://workspace.databricks.com/api/mcp");
  });

  describe("fromUCFunction", () => {
    it("creates server from UC function coordinates", async () => {
      const server = await DatabricksMCPServer.fromUCFunction("my_catalog", "my_schema", "my_function");

      expect(server.name).toBe("uc-function-my_catalog-my_schema-my_function");
      expect(server.url).toContain("/api/2.0/mcp/functions/my_catalog/my_schema/my_function");
    });

    it("creates server for all functions in schema when no function name provided", async () => {
      const server = await DatabricksMCPServer.fromUCFunction("my_catalog", "my_schema");

      expect(server.name).toBe("uc-functions-my_catalog-my_schema");
      expect(server.url).toContain("/api/2.0/mcp/functions/my_catalog/my_schema");
    });

    it("accepts additional options", async () => {
      const server = await DatabricksMCPServer.fromUCFunction("catalog", "schema", "func", {
        timeout: 60,
      });

      expect(server.timeout).toBe(60);
    });
  });

  describe("fromVectorSearch", () => {
    it("creates server from Vector Search coordinates", async () => {
      const server = await DatabricksMCPServer.fromVectorSearch("my_catalog", "my_schema", "my_index");

      expect(server.name).toBe("vector-search-my_catalog-my_schema-my_index");
      expect(server.url).toContain("/api/2.0/mcp/vector-search/my_catalog/my_schema/my_index");
    });

    it("creates server for all indexes in schema when no index name provided", async () => {
      const server = await DatabricksMCPServer.fromVectorSearch("my_catalog", "my_schema");

      expect(server.name).toBe("vector-search-my_catalog-my_schema");
      expect(server.url).toContain("/api/2.0/mcp/vector-search/my_catalog/my_schema");
    });

    it("accepts additional options", async () => {
      const server = await DatabricksMCPServer.fromVectorSearch("catalog", "schema", "index", {
        timeout: 120,
      });

      expect(server.timeout).toBe(120);
    });
  });

  describe("fromGenieSpace", () => {
    it("creates server from Genie Space ID", async () => {
      const server = await DatabricksMCPServer.fromGenieSpace("01ef19c578b21dc6af6e10983fb1e3f9");

      expect(server.name).toBe("genie-space-01ef19c578b21dc6af6e10983fb1e3f9");
      expect(server.url).toContain("/api/2.0/mcp/genie/01ef19c578b21dc6af6e10983fb1e3f9");
    });

    it("accepts additional options", async () => {
      const server = await DatabricksMCPServer.fromGenieSpace("my-space-id", {
        timeout: 90,
      });

      expect(server.timeout).toBe(90);
    });
  });

  describe("create", () => {
    it("creates server with path and resolves host from config", async () => {
      const server = await DatabricksMCPServer.create({
        name: "my-server",
        path: "/api/2.0/mcp/sql",
      });

      expect(server.name).toBe("my-server");
      expect(server.url).toContain("/api/2.0/mcp/sql");
    });

    it("normalizes path without leading slash", async () => {
      const server = await DatabricksMCPServer.create({
        name: "my-server",
        path: "api/2.0/mcp/sql",
      });

      expect(server.url).toContain("/api/2.0/mcp/sql");
    });

    it("accepts additional options", async () => {
      const server = await DatabricksMCPServer.create({
        name: "my-server",
        path: "/api/2.0/mcp/sql",
        timeout: 45,
      });

      expect(server.timeout).toBe(45);
    });
  });
});
