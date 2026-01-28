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

  it("converts to connection config with http transport", async () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
    });

    const config = await server.toConnectionConfig();

    expect(config.transport).toBe("http");
    expect(config.url).toBe("https://example.com/mcp");
  });

  it("converts timeout to milliseconds in connection config", async () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
      timeout: 30,
    });

    const config = await server.toConnectionConfig();

    expect(config.defaultToolTimeout).toBe(30000); // 30 seconds in ms
  });

  it("includes headers in connection config", async () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
      headers: { Authorization: "Bearer token123" },
    });

    const config = await server.toConnectionConfig();

    expect(config.headers).toEqual({ Authorization: "Bearer token123" });
  });

  it("excludes name from connection config", async () => {
    const server = new MCPServer({
      name: "test-server",
      url: "https://example.com/mcp",
    });

    const config = await server.toConnectionConfig();

    expect(config).not.toHaveProperty("name");
  });
});

describe("DatabricksMCPServer", () => {
  it("creates server with minimal config", () => {
    const server = new DatabricksMCPServer({
      name: "databricks-server",
      path: "/api/2.0/mcp/sql",
    });

    expect(server.name).toBe("databricks-server");
  });

  it("toConnectionConfig resolves URL and adds auth header", async () => {
    const server = new DatabricksMCPServer({
      name: "databricks-server",
      path: "/api/2.0/mcp/sql",
      auth: {
        host: "https://test.databricks.com",
        token: "test-token",
      },
    });

    const config = await server.toConnectionConfig();

    expect(config.transport).toBe("http");
    expect(config.url).toBe("https://test.databricks.com/api/2.0/mcp/sql");
    expect(config.headers).toHaveProperty("Authorization");
    expect(config.headers?.Authorization).toMatch(/^Bearer /);
  });

  describe("fromUCFunction", () => {
    it("creates server from UC function coordinates", () => {
      const server = DatabricksMCPServer.fromUCFunction("my_catalog", "my_schema", "my_function");

      expect(server.name).toBe("uc-function-my_catalog-my_schema-my_function");
      // URL is resolved lazily, so we can't check it here
    });

    it("creates server for all functions in schema when no function name provided", () => {
      const server = DatabricksMCPServer.fromUCFunction("my_catalog", "my_schema");

      expect(server.name).toBe("uc-functions-my_catalog-my_schema");
    });

    it("accepts additional options", () => {
      const server = DatabricksMCPServer.fromUCFunction("catalog", "schema", "func", {
        timeout: 60,
      });

      expect(server.timeout).toBe(60);
    });
  });

  describe("fromVectorSearch", () => {
    it("creates server from Vector Search coordinates", () => {
      const server = DatabricksMCPServer.fromVectorSearch("my_catalog", "my_schema", "my_index");

      expect(server.name).toBe("vector-search-my_catalog-my_schema-my_index");
    });

    it("creates server for all indexes in schema when no index name provided", () => {
      const server = DatabricksMCPServer.fromVectorSearch("my_catalog", "my_schema");

      expect(server.name).toBe("vector-search-my_catalog-my_schema");
    });

    it("accepts additional options", () => {
      const server = DatabricksMCPServer.fromVectorSearch("catalog", "schema", "index", {
        timeout: 120,
      });

      expect(server.timeout).toBe(120);
    });
  });

  describe("fromGenieSpace", () => {
    it("creates server from Genie Space ID", () => {
      const server = DatabricksMCPServer.fromGenieSpace("01ef19c578b21dc6af6e10983fb1e3f9");

      expect(server.name).toBe("genie-space-01ef19c578b21dc6af6e10983fb1e3f9");
    });

    it("accepts additional options", () => {
      const server = DatabricksMCPServer.fromGenieSpace("my-space-id", {
        timeout: 90,
      });

      expect(server.timeout).toBe(90);
    });
  });

  describe("constructor", () => {
    it("creates server with path (host resolved lazily)", () => {
      const server = new DatabricksMCPServer({
        name: "my-server",
        path: "/api/2.0/mcp/sql",
      });

      expect(server.name).toBe("my-server");
      // URL is resolved lazily in toConnectionConfigWithHeaders()
    });

    it("normalizes path without leading slash", () => {
      const server = new DatabricksMCPServer({
        name: "my-server",
        path: "api/2.0/mcp/sql",
      });

      expect(server.name).toBe("my-server");
    });

    it("accepts additional options", () => {
      const server = new DatabricksMCPServer({
        name: "my-server",
        path: "/api/2.0/mcp/sql",
        timeout: 45,
      });

      expect(server.timeout).toBe(45);
    });
  });
});
