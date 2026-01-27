/**
 * Integration tests for ChatDatabricks
 *
 * These tests require a real Databricks workspace connection.
 * Set the following environment variables:
 *   - DATABRICKS_HOST: Your workspace URL (e.g., https://your-workspace.databricks.com)
 *   - DATABRICKS_TOKEN: Your personal access token (or use other auth methods)
 */
import "dotenv/config";
import { config } from "dotenv";
config({ path: ".env.local" });

import { describe, it, expect, beforeAll } from "vitest";
import { AIMessageChunk, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { ChatDatabricks } from "../src/chat_models.js";

// Endpoint configurations to test
const endpointConfigs: { name: string; useResponsesApi: boolean; endpoint: string }[] = [
  {
    name: "chat-completions",
    useResponsesApi: false,
    endpoint: "databricks-claude-sonnet-4-5",
  },
  {
    name: "responses",
    useResponsesApi: true,
    endpoint: "databricks-gpt-5-2",
  },
];

// Skip all tests if no Databricks credentials are configured
const hasCredentials =
  process.env.DATABRICKS_HOST ||
  process.env.DATABRICKS_TOKEN ||
  (process.env.CLIENT_ID && process.env.CLIENT_SECRET);

describe.skipIf(!hasCredentials)("ChatDatabricks Integration Tests", () => {
  describe.each(endpointConfigs)("$name endpoint", ({ useResponsesApi, endpoint }) => {
    let model: ChatDatabricks;

    beforeAll(() => {
      model = new ChatDatabricks({
        endpoint,
        useResponsesApi,
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

        const response1 = await modelWithTools.invoke([new HumanMessage("What time is it?")]);

        if (response1.tool_calls && response1.tool_calls.length > 0) {
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

        const finalChunk = chunks[chunks.length - 1];
        expect(
          finalChunk.tool_calls !== undefined || finalChunk.tool_call_chunks !== undefined
        ).toBeTruthy();
      });
    });

    describe("Parameters", () => {
      it("respects temperature parameter", async () => {
        const coldModel = new ChatDatabricks({
          endpoint,
          useResponsesApi,
          temperature: 0,
          maxTokens: 50,
        });

        const responses = await Promise.all([
          coldModel.invoke([new HumanMessage("What is 2+2? Just give the number.")]),
          coldModel.invoke([new HumanMessage("What is 2+2? Just give the number.")]),
        ]);

        expect(responses[0].content as string).toContain("4");
        expect(responses[1].content as string).toContain("4");
      });

      it("respects maxTokens parameter", async () => {
        const shortModel = new ChatDatabricks({
          endpoint,
          useResponsesApi,
          maxTokens: 16,
        });

        const response = await shortModel.invoke([
          new HumanMessage("Write a very long essay about the history of computing"),
        ]);

        const content = response.content as string;
        expect(content.split(/\s+/).length).toBeLessThan(50);
      });

      it
        // Responses API doesn't support stop sequences
        // See https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#responses-api-request
        .skipIf(useResponsesApi)("respects stop sequences", async () => {
        const modelWithStop = new ChatDatabricks({
          endpoint,
          useResponsesApi,
          maxTokens: 100,
          stop: ["5"],
        });

        const response = await modelWithStop.invoke([
          new HumanMessage("Count from 1 to 10, one number per line"),
        ]);

        const content = response.content as string;
        expect(content).not.toContain("6");
      });
    });

    describe("Error Handling", () => {
      it("throws on invalid endpoint", async () => {
        const badModel = new ChatDatabricks({
          endpoint: "nonexistent-endpoint-12345",
          useResponsesApi,
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
        expect(content).toBeTruthy();
        expect(/[\u3000-\u9fff\uac00-\ud7af]/.test(content)).toBeTruthy();
      });

      it("handles emoji in input and output", async () => {
        const response = await model.invoke([new HumanMessage("Reply with only: 👋🌍")]);

        const content = response.content as string;
        expect(content).toBeTruthy();
      });
    });
  });
});
