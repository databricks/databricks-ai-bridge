/**
 * Unit tests for ChatDatabricks
 */

import { describe, it, expect } from "vitest";
import { HumanMessage, AIMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { Config } from "@databricks/sdk-experimental";
import { z } from "zod";
import { DynamicStructuredTool } from "@langchain/core/tools";

import { convertLangChainToModelMessages } from "../src/utils/messages.js";
import { convertToAISDKToolSet } from "../src/utils/tools.js";

describe("Message Conversion", () => {
  describe("convertLangChainToModelMessages", () => {
    it("converts HumanMessage to user role", () => {
      const messages = [new HumanMessage("Hello")];
      const result = convertLangChainToModelMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0].role).toBe("user");
      expect(result[0].content).toEqual([{ type: "text", text: "Hello" }]);
    });

    it("converts SystemMessage to system role", () => {
      const messages = [new SystemMessage("You are a helpful assistant")];
      const result = convertLangChainToModelMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0].role).toBe("system");
      expect(result[0].content).toBe("You are a helpful assistant");
    });

    it("converts AIMessage to assistant role", () => {
      const messages = [new AIMessage("Hello, how can I help?")];
      const result = convertLangChainToModelMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0].role).toBe("assistant");
      expect(result[0].content).toEqual([{ type: "text", text: "Hello, how can I help?" }]);
    });

    it("converts AIMessage with tool calls", () => {
      const messages = [
        new AIMessage({
          content: "",
          tool_calls: [
            {
              id: "call_123",
              name: "get_weather",
              args: { location: "San Francisco" },
            },
          ],
        }),
      ];
      const result = convertLangChainToModelMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "call_123",
            toolName: "get_weather",
            input: { location: "San Francisco" },
          },
        ],
      });
    });

    it("converts ToolMessage to tool role", () => {
      const messages = [
        new ToolMessage({
          content: '{"temperature": 72}',
          tool_call_id: "call_123",
        }),
      ];
      const result = convertLangChainToModelMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_123",
            toolName: "unknown",
            output: {
              type: "json",
              value: { temperature: 72 },
            },
          },
        ],
      });
    });

    it("converts ToolMessage with plain text to text output", () => {
      const messages = [
        new ToolMessage({
          content: "The weather is sunny",
          tool_call_id: "call_456",
        }),
      ];
      const result = convertLangChainToModelMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_456",
            toolName: "unknown",
            output: {
              type: "text",
              value: "The weather is sunny",
            },
          },
        ],
      });
    });

    it("handles multi-turn conversation", () => {
      const messages = [
        new SystemMessage("You are a helpful assistant"),
        new HumanMessage("What's the weather?"),
        new AIMessage({
          content: "",
          tool_calls: [{ id: "call_1", name: "get_weather", args: { location: "NYC" } }],
        }),
        new ToolMessage({ content: "72°F", tool_call_id: "call_1" }),
        new AIMessage("The weather in NYC is 72°F."),
      ];
      const result = convertLangChainToModelMessages(messages);

      expect(result).toHaveLength(5);
      expect(result[0].role).toBe("system");
      expect(result[1].role).toBe("user");
      expect(result[2].role).toBe("assistant");
      expect(result[3].role).toBe("tool");
      expect(result[4].role).toBe("assistant");
    });

    it("throws error on image content", () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: "text", text: "What's in this image?" },
            { type: "image_url", image_url: { url: "https://example.com/image.png" } },
          ],
        }),
      ];

      expect(() => convertLangChainToModelMessages(messages)).toThrow(
        "Image content is not yet supported in ChatDatabricks"
      );
    });
  });
});

describe("Tool Conversion", () => {
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

      // ToolSet is a Record<string, Tool>
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

      // ToolSet is a Record<string, Tool>
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

    it("converts StructuredToolParams (minimal tool definition)", () => {
      // StructuredToolParams has: name, schema, extras?, description?
      const tools = [
        {
          name: "lookup",
          description: "Look up information",
          schema: {
            type: "object",
            properties: {
              id: { type: "string" },
            },
          },
        },
      ];

      const result = convertToAISDKToolSet(tools);

      expect(result).toHaveProperty("lookup");
      expect(result.lookup.description).toBe("Look up information");
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

    it("converts multiple DynamicStructuredTools", () => {
      const searchTool = new DynamicStructuredTool({
        name: "search",
        description: "Search the web",
        schema: z.object({
          query: z.string(),
        }),
        func: async () => "results",
      });

      const calculatorTool = new DynamicStructuredTool({
        name: "calculator",
        description: "Do math",
        schema: z.object({
          expression: z.string(),
        }),
        func: async () => "42",
      });

      const result = convertToAISDKToolSet([searchTool, calculatorTool]);

      expect(result).toHaveProperty("search");
      expect(result).toHaveProperty("calculator");
      expect(result.search.description).toBe("Search the web");
      expect(result.calculator.description).toBe("Do math");
    });
  });
});

describe("ChatDatabricks", () => {
  describe("constructor validation", () => {
    it("creates model with Config object", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      const config = new Config({
        host: "https://test.databricks.com",
        token: "test-token",
      });

      const model = new ChatDatabricks({
        endpoint: "test-endpoint",
        config,
      });

      expect(model.endpoint).toBe("test-endpoint");
    });

    it("supports endpoint types", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      const config = new Config({
        host: "https://test.databricks.com",
        token: "test-token",
      });

      const fmapiModel = new ChatDatabricks({
        endpoint: "test-endpoint",
        endpointType: "fmapi",
        config,
      });

      const chatAgentModel = new ChatDatabricks({
        endpoint: "test-agent",
        endpointType: "chat-agent",
        config,
      });

      const responsesAgentModel = new ChatDatabricks({
        endpoint: "test-responses",
        endpointType: "responses-agent",
        config,
      });

      expect(fmapiModel.endpointType).toBe("fmapi");
      expect(chatAgentModel.endpointType).toBe("chat-agent");
      expect(responsesAgentModel.endpointType).toBe("responses-agent");
    });

    it("defaults to fmapi endpoint type", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      const config = new Config({
        host: "https://test.databricks.com",
        token: "test-token",
      });

      const model = new ChatDatabricks({
        endpoint: "test-endpoint",
        config,
      });

      expect(model.endpointType).toBe("fmapi");
    });

    it("creates default Config if not provided", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      // This will use environment variables or ~/.databrickscfg
      const model = new ChatDatabricks({
        endpoint: "test-endpoint",
      });

      expect(model.endpoint).toBe("test-endpoint");
      expect(model.endpointType).toBe("fmapi");
    });
  });
});
