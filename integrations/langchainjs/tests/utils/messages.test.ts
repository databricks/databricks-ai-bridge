/**
 * Unit tests for message conversion utilities
 */

import { describe, it, expect } from "vitest";
import { HumanMessage, AIMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";

import {
  convertLangChainToModelMessages,
  convertGenerateTextResultToChatResult,
} from "../../src/utils/messages.js";
import { GenerateTextResult, ToolSet } from "ai";

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

  it("handles HumanMessage with array of text parts", () => {
    const messages = [
      new HumanMessage({
        content: [
          { type: "text", text: "First part" },
          { type: "text", text: "Second part" },
        ],
      }),
    ];
    const result = convertLangChainToModelMessages(messages);

    expect(result).toHaveLength(1);
    expect(result[0].content).toEqual([
      { type: "text", text: "First part" },
      { type: "text", text: "Second part" },
    ]);
  });

  it("handles AIMessage with both text and tool calls", () => {
    const messages = [
      new AIMessage({
        content: "Let me check the weather for you.",
        tool_calls: [
          {
            id: "call_123",
            name: "get_weather",
            args: { location: "NYC" },
          },
        ],
      }),
    ];
    const result = convertLangChainToModelMessages(messages);

    expect(result).toHaveLength(1);
    expect(result[0].content).toEqual([
      { type: "text", text: "Let me check the weather for you." },
      {
        type: "tool-call",
        toolCallId: "call_123",
        toolName: "get_weather",
        input: { location: "NYC" },
      },
    ]);
  });

  it("generates tool call ID when missing", () => {
    const messages = [
      new AIMessage({
        content: "",
        tool_calls: [
          {
            name: "get_weather",
            args: { location: "NYC" },
          },
        ],
      }),
    ];
    const result = convertLangChainToModelMessages(messages);

    expect(result[0].content).toEqual([
      {
        type: "tool-call",
        toolCallId: "tool-call-get_weather",
        toolName: "get_weather",
        input: { location: "NYC" },
      },
    ]);
  });
});

describe("convertGenerateTextResultToChatResult", () => {
  it("converts basic text result", () => {
    const result = {
      text: "Hello, world!",
      finishReason: "stop" as const,
      usage: {
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: 15,
      },
      response: {
        id: "test-id",
        modelId: "test-model",
        timestamp: new Date(),
        messages: [],
      },
    } satisfies Partial<GenerateTextResult<ToolSet, unknown>>;

    const chatResult = convertGenerateTextResultToChatResult(result as any);

    expect(chatResult.generations).toHaveLength(1);
    expect(chatResult.generations[0].text).toBe("Hello, world!");
    expect(chatResult.generations[0].message.content).toBe("Hello, world!");
    expect(chatResult.generations[0].generationInfo?.finish_reason).toBe("stop");
    expect(chatResult.llmOutput?.tokenUsage).toEqual({
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    });
  });

  it("converts result with tool calls", () => {
    const result = {
      text: "",
      finishReason: "tool-calls" as const,
      toolCalls: [
        {
          type: "tool-call",
          toolCallId: "call_123",
          toolName: "get_weather",
          input: { location: "San Francisco" },
        },
      ],
      usage: {
        inputTokens: 20,
        outputTokens: 10,
        totalTokens: 30,
      },
      response: {
        id: "test-id",
        modelId: "test-model",
        timestamp: new Date(),
        messages: [],
      },
      reasoning: undefined,
      files: [],
      sources: [],
    } satisfies Partial<GenerateTextResult<ToolSet, unknown>>;

    const chatResult = convertGenerateTextResultToChatResult(result as any);

    expect(chatResult.generations).toHaveLength(1);
    const message = chatResult.generations[0].message;
    expect(AIMessage.isInstance(message)).toBe(true);
    if (!AIMessage.isInstance(message)) return;
    expect(message.tool_calls).toHaveLength(1);
    expect(message.tool_calls?.[0]).toEqual({
      id: "call_123",
      name: "get_weather",
      args: { location: "San Francisco" },
      type: "tool_call",
    });
  });

  it("handles result without usage information", () => {
    const result = {
      text: "Response without usage",
      finishReason: "stop" as const,
      usage: undefined,
      response: {
        id: "test-id",
        modelId: "test-model",
        timestamp: new Date(),
        messages: [],
      },
      reasoning: undefined,
      files: [],
      sources: [],
    } satisfies Partial<GenerateTextResult<ToolSet, unknown>>;

    const chatResult = convertGenerateTextResultToChatResult(result as any);

    expect(chatResult.generations).toHaveLength(1);
    expect(chatResult.generations[0].text).toBe("Response without usage");
    expect(chatResult.generations[0].message.content).toBe("Response without usage");
    expect(chatResult.generations[0].generationInfo?.finish_reason).toBe("stop");
    expect(chatResult.llmOutput?.tokenUsage).toBeUndefined();
  });
});
