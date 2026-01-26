/**
 * Unit tests for tool conversion utilities
 */

import { describe, it, expect } from "vitest";
import { z } from "zod";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { DATABRICKS_TOOL_CALL_ID } from "@databricks/ai-sdk-provider";

import { convertToAISDKToolSet, getToolNameFromAiSDKTool } from "../../src/utils/tools.js";

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

  it("includes Databricks tool definition in result", () => {
    const tools = [
      {
        name: "my_tool",
        description: "My tool",
      },
    ];

    const result = convertToAISDKToolSet(tools);

    expect(result).toHaveProperty(DATABRICKS_TOOL_CALL_ID);
    expect(result).toHaveProperty("my_tool");
  });

  it("handles empty tools array", () => {
    const result = convertToAISDKToolSet([]);

    // Should still have the Databricks tool
    expect(result).toHaveProperty(DATABRICKS_TOOL_CALL_ID);
    expect(Object.keys(result)).toHaveLength(1);
  });
});

describe("getToolNameFromAiSDKTool", () => {
  it("returns toolName for regular tools", () => {
    const toolCall = {
      type: "tool-call" as const,
      toolCallId: "call_123",
      toolName: "get_weather",
      input: { location: "NYC" },
    };

    const result = getToolNameFromAiSDKTool(toolCall);

    expect(result).toBe("get_weather");
  });

  it("extracts toolName from Databricks provider metadata", () => {
    const toolCall = {
      type: "tool-call" as const,
      toolCallId: "call_456",
      toolName: DATABRICKS_TOOL_CALL_ID,
      input: { location: "SF" },
      providerMetadata: {
        databricks: {
          toolName: "actual_tool_name",
        },
      },
    };

    const result = getToolNameFromAiSDKTool(toolCall);

    expect(result).toBe("actual_tool_name");
  });

  it("falls back to toolName when Databricks metadata is missing", () => {
    const toolCall = {
      type: "tool-call" as const,
      toolCallId: "call_789",
      toolName: DATABRICKS_TOOL_CALL_ID,
      input: {},
      providerMetadata: {},
    };

    const result = getToolNameFromAiSDKTool(toolCall);

    expect(result).toBe(DATABRICKS_TOOL_CALL_ID);
  });
});
