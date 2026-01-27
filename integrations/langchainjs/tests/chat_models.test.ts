/**
 * Unit tests for ChatDatabricks
 */

import { describe, it, expect } from "vitest";

describe("ChatDatabricks", () => {
  describe("constructor validation", () => {
    it("creates model with auth object", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      const model = new ChatDatabricks({
        endpoint: "test-endpoint",
        auth: {
          host: "https://test.databricks.com",
          token: "test-token",
        },
      });

      expect(model.endpoint).toBe("test-endpoint");
      expect(model.useResponsesApi).toBeUndefined();
    });

    it("supports useResponsesApi flag", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      const auth = {
        host: "https://test.databricks.com",
        token: "test-token",
      };

      const chatCompletionsModel = new ChatDatabricks({
        endpoint: "test-endpoint",
        useResponsesApi: false,
        auth,
      });

      const responsesModel = new ChatDatabricks({
        endpoint: "test-responses",
        useResponsesApi: true,
        auth,
      });

      expect(chatCompletionsModel.useResponsesApi).toBe(false);
      expect(responsesModel.useResponsesApi).toBe(true);
    });

    it("accepts model parameters", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      const model = new ChatDatabricks({
        endpoint: "test-endpoint",
        auth: {
          host: "https://test.databricks.com",
          token: "test-token",
        },
        temperature: 0.7,
        maxTokens: 1000,
        stop: ["\n\n"],
      });

      expect(model.temperature).toBe(0.7);
      expect(model.maxTokens).toBe(1000);
      expect(model.stop).toEqual(["\n\n"]);
    });

    it("returns correct llm type", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      const model = new ChatDatabricks({
        endpoint: "test-endpoint",
        auth: {
          host: "https://test.databricks.com",
          token: "test-token",
        },
      });

      expect(model._llmType()).toBe("chat-databricks");
    });

    it("returns identifying params", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      const model = new ChatDatabricks({
        endpoint: "test-endpoint",
        useResponsesApi: true,
        auth: {
          host: "https://test.databricks.com",
          token: "test-token",
        },
        temperature: 0.5,
        maxTokens: 500,
      });

      const params = model.identifyingParams;

      expect(params.endpoint).toBe("test-endpoint");
      expect(params.useResponsesApi).toBe(true);
      expect(params.temperature).toBe(0.5);
      expect(params.maxTokens).toBe(500);
    });

    it("has correct lc_name", async () => {
      const { ChatDatabricks } = await import("../src/chat_models.js");

      expect(ChatDatabricks.lc_name()).toBe("ChatDatabricks");
    });
  });
});
