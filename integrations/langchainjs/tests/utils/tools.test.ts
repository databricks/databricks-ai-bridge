/**
 * Unit tests for tool conversion utilities
 */

import { describe, it, expect } from "vitest";
import { z } from "zod";
import { DynamicStructuredTool } from "@langchain/core/tools";

import { convertToAISDKToolSet } from "../../src/utils/tools.js";

describe("convertToAISDKToolSet", () => {
  it("converts OpenAI format tools (ToolDefinition)", () => {
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "get_weather",
          description: "Get the weather",
          parameters: {
            type: "object",
            properties: { location: { type: "string" } },
          },
        },
      },
    ];

    const result = convertToAISDKToolSet(tools);

    expect(result).toHaveProperty("get_weather");
    expect(result.get_weather.description).toBe("Get the weather");
  });

  it("converts plain object with parameters (Record<string, any>)", () => {
    const tools = [
      {
        name: "search",
        description: "Search the web",
        parameters: {
          type: "object",
          properties: { query: { type: "string" } },
        },
      },
    ];

    const result = convertToAISDKToolSet(tools);

    expect(result).toHaveProperty("search");
    expect(result.search.description).toBe("Search the web");
  });

  it("converts plain object with schema (Record<string, any>)", () => {
    const tools = [
      {
        name: "calculate",
        description: "Perform calculation",
        schema: {
          type: "object",
          properties: {
            expression: { type: "string" },
          },
        },
      },
    ];

    const result = convertToAISDKToolSet(tools);

    expect(result).toHaveProperty("calculate");
    expect(result.calculate.description).toBe("Perform calculation");
  });

  it("handles tools without description", () => {
    const tools = [
      {
        name: "ping",
        parameters: {
          type: "object",
          properties: {},
        },
      },
    ];

    const result = convertToAISDKToolSet(tools);

    expect(result).toHaveProperty("ping");
    expect(result.ping.description).toBeUndefined();
  });

  it("handles tools without schema/parameters", () => {
    const tools = [
      {
        name: "noop",
        description: "Does nothing",
      },
    ];

    const result = convertToAISDKToolSet(tools);

    expect(result).toHaveProperty("noop");
    expect(result.noop.description).toBe("Does nothing");
  });

  it("converts multiple tools of different types", () => {
    const tools = [
      // OpenAI format
      {
        type: "function" as const,
        function: {
          name: "tool_a",
          description: "Tool A",
          parameters: { type: "object", properties: {} },
        },
      },
      // Plain object with parameters
      {
        name: "tool_b",
        description: "Tool B",
        parameters: { type: "object", properties: {} },
      },
      // Plain object with schema
      {
        name: "tool_c",
        description: "Tool C",
        schema: { type: "object", properties: {} },
      },
    ];

    const result = convertToAISDKToolSet(tools);

    expect(result).toHaveProperty("tool_a");
    expect(result).toHaveProperty("tool_b");
    expect(result).toHaveProperty("tool_c");
    expect(result.tool_a.description).toBe("Tool A");
    expect(result.tool_b.description).toBe("Tool B");
    expect(result.tool_c.description).toBe("Tool C");
  });

  it("throws error for unsupported tool type", () => {
    const tools = [
      "not a tool object", // string is not a valid tool
    ];

    expect(() => convertToAISDKToolSet(tools as any)).toThrow("Unsupported tool type");
  });

  it("converts DynamicStructuredTool with Zod schema", () => {
    const weatherTool = new DynamicStructuredTool({
      name: "get_weather",
      description: "Get the weather for a location",
      schema: z.object({
        location: z.string().describe("The city to get weather for"),
        unit: z.enum(["celsius", "fahrenheit"]).optional(),
      }),
      func: async () => "sunny",
    });

    const result = convertToAISDKToolSet([weatherTool]);

    expect(result).toHaveProperty("get_weather");
    expect(result.get_weather.description).toBe("Get the weather for a location");
  });

  it("converts single tool correctly", () => {
    const tools = [
      {
        name: "my_tool",
        description: "My tool",
      },
    ];

    const result = convertToAISDKToolSet(tools);

    expect(result).toHaveProperty("my_tool");
    expect(Object.keys(result)).toHaveLength(1);
  });

  it("handles empty tools array", () => {
    const result = convertToAISDKToolSet([]);

    expect(Object.keys(result)).toHaveLength(0);
  });
});
