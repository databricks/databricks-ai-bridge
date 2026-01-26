/**
 * Integration tests for ChatDatabricks
 *
 * These tests require a real Databricks workspace connection.
 * Set the following environment variables:
 *   - DATABRICKS_HOST: Your workspace URL (e.g., https://your-workspace.databricks.com)
 *   - DATABRICKS_TOKEN: Your personal access token (or use other auth methods)
 *   - TEST_ENDPOINT_NAME: Model serving endpoint to test against (default: databricks-meta-llama-3-3-70b-instruct)
 */

import { describe, it, expect, beforeAll } from "vitest";
import { AIMessageChunk, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { ChatDatabricks } from "../src/chat_models.js";

const ENDPOINT_NAME = 'databricks-claude-sonnet-4-5';
const ENDPOINT_API = 'chat-completions';

// Skip all tests if no Databricks credentials are configured
const hasCredentials = process.env.DATABRICKS_HOST || process.env.DATABRICKS_TOKEN;

describe.skipIf(!hasCredentials)("ChatDatabricks Integration Tests", () => {
  let model: ChatDatabricks;

  beforeAll(() => {
    model = new ChatDatabricks({
      endpoint: ENDPOINT_NAME,
      endpointAPI: ENDPOINT_API,
      maxTokens: 100,
    });
  });

  describe("Basic Chat Completion", () => {
    it("completes a simple message", async () => {
      const response = await model.invoke([new HumanMessage("Say hello in exactly 3 words")]);

      expect(response.content).toBeTruthy();
      expect(typeof response.content).toBe("string");
    });

    it("handles system message", async () => {
      const response = await model.invoke([
        new SystemMessage("You are a pirate. Always respond like a pirate."),
        new HumanMessage("How are you?"),
      ]);

      expect(response.content).toBeTruthy();
      // Response should have some pirate-like characteristics
      const content = response.content as string;
      expect(content.length).toBeGreaterThan(0);
    });

    it("handles multi-turn conversation", async () => {
      const response = await model.invoke([
        new HumanMessage("My name is Alice"),
        new HumanMessage("What is my name?"),
      ]);

      const content = response.content as string;
      expect(content.toLowerCase()).toContain("alice");
    });
  });

  describe("Streaming", () => {
    it("streams response chunks", async () => {
      const chunks: string[] = [];

      const stream = await model.stream([new HumanMessage("Count from 1 to 5")]);

      for await (const chunk of stream) {
        if (chunk.content) {
          chunks.push(chunk.content as string);
        }
      }

      expect(chunks.length).toBeGreaterThan(1);
      const fullResponse = chunks.join("");
      expect(fullResponse).toBeTruthy();
    });

    it("can abort streaming", async () => {
      const controller = new AbortController();
      const chunks: string[] = [];

      const stream = await model.stream([new HumanMessage("Write a very long story")], {
        signal: controller.signal,
      });

      let count = 0;
      try {
        for await (const chunk of stream) {
          chunks.push(chunk.content as string);
          count++;
          if (count >= 3) {
            controller.abort();
          }
        }
      } catch (e) {
        expect((e as Error).message).toContain("abort");
      }

      // Should have received some chunks before abort
      expect(chunks.length).toBeGreaterThan(0);
    });
  });

  describe("Tool Calling", () => {
    it("calls a tool when appropriate", async () => {
      const modelWithTools = model.bindTools([
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Get the current weather for a location",
            parameters: {
              type: "object",
              properties: {
                location: {
                  type: "string",
                  description: "The city and state, e.g. San Francisco, CA",
                },
              },
              required: ["location"],
            },
          },
        },
      ]);

      const response = await modelWithTools.invoke([
        new HumanMessage("What's the weather like in San Francisco?"),
      ]);

      // The model should call the get_weather tool
      expect(response.tool_calls).toBeDefined();
      expect(response.tool_calls?.length).toBeGreaterThan(0);
      expect(response.tool_calls?.[0].name).toBe("get_weather");
    });

    it("handles tool response in conversation", async () => {
      const modelWithTools = model.bindTools([
        {
          type: "function",
          function: {
            name: "get_time",
            description: "Get the current time",
            parameters: {
              type: "object",
              properties: {},
            },
          },
        },
      ]);

      // First call - model should request tool
      const response1 = await modelWithTools.invoke([new HumanMessage("What time is it?")]);

      if (response1.tool_calls && response1.tool_calls.length > 0) {
        // Simulate tool response
        const toolResponse = await modelWithTools.invoke([
          new HumanMessage("What time is it?"),
          response1,
          new ToolMessage({
            content: "The current time is 3:00 PM",
            tool_call_id: response1.tool_calls[0].id!,
          }),
        ]);

        expect(toolResponse.content).toBeTruthy();
        const content = toolResponse.content as string;
        expect(content.toLowerCase()).toMatch(/3|three|pm|afternoon/);
      }
    });

    it("streams with tool calls", async () => {
      const modelWithTools = model.bindTools([
        {
          type: "function",
          function: {
            name: "calculate",
            description: "Perform a calculation",
            parameters: {
              type: "object",
              properties: {
                expression: { type: "string" },
              },
              required: ["expression"],
            },
          },
        },
      ]);

      const chunks: AIMessageChunk[] = [];
      const stream = await modelWithTools.stream([new HumanMessage("Calculate 2 + 2")]);

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);

      // Find the final chunk with tool calls
      const finalChunk = chunks[chunks.length - 1];
      // Tool calls may be accumulated in the final chunk
      expect(
        finalChunk.tool_calls !== undefined || finalChunk.tool_call_chunks !== undefined
      ).toBeTruthy();
    });
  });

  describe("Parameters", () => {
    it("respects temperature parameter", async () => {
      const coldModel = new ChatDatabricks({
        endpoint: ENDPOINT_NAME,
        endpointAPI: ENDPOINT_API,
        temperature: 0,
        maxTokens: 50,
      });

      // With temperature 0, responses should be more deterministic
      const responses = await Promise.all([
        coldModel.invoke([new HumanMessage("What is 2+2? Just give the number.")]),
        coldModel.invoke([new HumanMessage("What is 2+2? Just give the number.")]),
      ]);

      // Both responses should contain "4"
      expect(responses[0].content as string).toContain("4");
      expect(responses[1].content as string).toContain("4");
    });

    it("respects maxTokens parameter", async () => {
      const shortModel = new ChatDatabricks({
        endpoint: ENDPOINT_NAME,
        endpointAPI: ENDPOINT_API,
        maxTokens: 10,
      });

      const response = await shortModel.invoke([
        new HumanMessage("Write a very long essay about the history of computing"),
      ]);

      // Response should be truncated due to low max_tokens
      const content = response.content as string;
      expect(content.split(/\s+/).length).toBeLessThan(50);
    });

    it("respects stop sequences", async () => {
      const modelWithStop = new ChatDatabricks({
        endpoint: ENDPOINT_NAME,
        endpointAPI: ENDPOINT_API,
        maxTokens: 100,
        stop: ["5"],
      });

      const response = await modelWithStop.invoke([
        new HumanMessage("Count from 1 to 10, one number per line"),
      ]);

      const content = response.content as string;
      // Should stop before or at 5
      expect(content).not.toContain("6");
    });
  });

  describe("Error Handling", () => {
    it("throws on invalid endpoint", async () => {
      const badModel = new ChatDatabricks({
        endpoint: "nonexistent-endpoint-12345",
        endpointAPI: ENDPOINT_API,
      });

      await expect(badModel.invoke([new HumanMessage("Hello")])).rejects.toThrow();
    });
  });

  describe("UTF-8 Encoding", () => {
    it("handles unicode content correctly", async () => {
      const response = await model.invoke([
        new HumanMessage(
          "Translate 'hello' to Japanese, Chinese, and Korean. Just give the translations."
        ),
      ]);

      const content = response.content as string;
      // Should contain non-ASCII characters
      expect(content).toBeTruthy();
      // Check for presence of CJK characters
      expect(/[\u3000-\u9fff\uac00-\ud7af]/.test(content)).toBeTruthy();
    });

    it("handles emoji in input and output", async () => {
      const response = await model.invoke([new HumanMessage("Reply with only: 👋🌍")]);

      const content = response.content as string;
      expect(content).toBeTruthy();
    });
  });
});
