/**
 * Unit tests for ChatDatabricks
 */

import { describe, it, expect } from "vitest";
import { HumanMessage, AIMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";

import { convertMessagesToOpenAI, convertResponseToChatResult } from "../utils/messages.js";
import { convertToOpenAITools, convertToolChoice } from "../utils/tools.js";
import type { ChatCompletionResponse, ChatCompletionTool } from "../types.js";

describe("Message Conversion", () => {
  describe("convertMessagesToOpenAI", () => {
    it("converts HumanMessage to user role", () => {
      const messages = [new HumanMessage("Hello")];
      const result = convertMessagesToOpenAI(messages);

      expect(result).toEqual([{ role: "user", content: "Hello" }]);
    });

    it("converts SystemMessage to system role", () => {
      const messages = [new SystemMessage("You are a helpful assistant")];
      const result = convertMessagesToOpenAI(messages);

      expect(result).toEqual([{ role: "system", content: "You are a helpful assistant" }]);
    });

    it("converts AIMessage to assistant role", () => {
      const messages = [new AIMessage("Hello, how can I help?")];
      const result = convertMessagesToOpenAI(messages);

      expect(result).toEqual([{ role: "assistant", content: "Hello, how can I help?" }]);
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
      const result = convertMessagesToOpenAI(messages);

      expect(result).toEqual([
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "get_weather",
                arguments: '{"location":"San Francisco"}',
              },
            },
          ],
        },
      ]);
    });

    it("converts ToolMessage to tool role", () => {
      const messages = [
        new ToolMessage({
          content: '{"temperature": 72}',
          tool_call_id: "call_123",
        }),
      ];
      const result = convertMessagesToOpenAI(messages);

      expect(result).toEqual([
        {
          role: "tool",
          content: '{"temperature": 72}',
          tool_call_id: "call_123",
        },
      ]);
    });

    it("handles multi-turn conversation", () => {
      const messages = [
        new SystemMessage("You are a helpful assistant"),
        new HumanMessage("What's the weather?"),
        new AIMessage({
          content: "",
          tool_calls: [
            { id: "call_1", name: "get_weather", args: { location: "NYC" } },
          ],
        }),
        new ToolMessage({ content: "72°F", tool_call_id: "call_1" }),
        new AIMessage("The weather in NYC is 72°F."),
      ];
      const result = convertMessagesToOpenAI(messages);

      expect(result).toHaveLength(5);
      expect(result[0].role).toBe("system");
      expect(result[1].role).toBe("user");
      expect(result[2].role).toBe("assistant");
      expect(result[3].role).toBe("tool");
      expect(result[4].role).toBe("assistant");
    });
  });

  describe("convertResponseToChatResult", () => {
    it("converts simple response", () => {
      const response: ChatCompletionResponse = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1234567890,
        model: "llama-3-70b",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Hello!",
            },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
      };

      const result = convertResponseToChatResult(response);

      expect(result.generations).toHaveLength(1);
      expect(result.generations[0].text).toBe("Hello!");
      expect(result.generations[0].message.content).toBe("Hello!");
      expect(result.llmOutput?.tokenUsage).toEqual({
        promptTokens: 10,
        completionTokens: 5,
        totalTokens: 15,
      });
    });

    it("converts response with tool calls", () => {
      const response: ChatCompletionResponse = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1234567890,
        model: "llama-3-70b",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_abc",
                  type: "function",
                  function: {
                    name: "get_weather",
                    arguments: '{"location":"Boston"}',
                  },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      };

      const result = convertResponseToChatResult(response);

      expect(result.generations).toHaveLength(1);
      expect(result.generations[0].message.tool_calls).toEqual([
        {
          id: "call_abc",
          name: "get_weather",
          args: { location: "Boston" },
          type: "tool_call",
        },
      ]);
    });
  });
});

describe("Tool Conversion", () => {
  describe("convertToOpenAITools", () => {
    it("passes through OpenAI format tools", () => {
      const tools: ChatCompletionTool[] = [
        {
          type: "function",
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

      const result = convertToOpenAITools(tools);
      expect(result).toEqual(tools);
    });

    it("converts plain object tool definitions", () => {
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

      const result = convertToOpenAITools(tools);
      expect(result).toEqual([
        {
          type: "function",
          function: {
            name: "search",
            description: "Search the web",
            parameters: {
              type: "object",
              properties: { query: { type: "string" } },
            },
          },
        },
      ]);
    });
  });

  describe("convertToolChoice", () => {
    it("returns undefined for undefined input", () => {
      expect(convertToolChoice(undefined)).toBeUndefined();
    });

    it("passes through auto/none/required", () => {
      expect(convertToolChoice("auto")).toBe("auto");
      expect(convertToolChoice("none")).toBe("none");
      expect(convertToolChoice("required")).toBe("required");
    });

    it("converts any to required", () => {
      expect(convertToolChoice("any")).toBe("required");
    });

    it("converts tool name to function choice", () => {
      expect(convertToolChoice("get_weather")).toEqual({
        type: "function",
        function: { name: "get_weather" },
      });
    });

    it("converts object format", () => {
      const choice = { type: "function", function: { name: "search" } };
      expect(convertToolChoice(choice)).toEqual({
        type: "function",
        function: { name: "search" },
      });
    });
  });
});

describe("ChatDatabricks", () => {
  // Note: Full ChatDatabricks tests with mocked HTTP client would go here
  // For now, we test the utility functions that it depends on

  describe("constructor validation", () => {
    it("requires endpoint parameter", async () => {
      // Dynamic import to avoid issues with mocking
      const { ChatDatabricks } = await import("../chat_models.js");

      // This should not throw during construction
      // (actual endpoint validation happens at request time)
      const model = new ChatDatabricks({
        endpoint: "test-endpoint",
        host: "https://test.databricks.com",
        token: "test-token",
      });

      expect(model.endpoint).toBe("test-endpoint");
    });
  });
});
