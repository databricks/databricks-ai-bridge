import { describe, expect, test } from "vitest";
import type { FunctionTool } from "../../agent-plugin/function-tool";
import {
  functionToolsToStructuredTools,
  functionToolToStructuredTool,
  isFunctionTool,
} from "../../agent-plugin/function-tool";

const weatherTool: FunctionTool = {
  type: "function",
  name: "get_weather",
  description: "Get the current weather for a location",
  parameters: {
    type: "object",
    properties: {
      location: { type: "string", description: "City name" },
    },
    required: ["location"],
  },
  execute: async ({ location }) => `Sunny in ${location}`,
};

const noParamsTool: FunctionTool = {
  type: "function",
  name: "get_time",
  description: "Get current time",
  execute: async () => new Date().toISOString(),
};

describe("isFunctionTool", () => {
  test("returns true for valid FunctionTool", () => {
    expect(isFunctionTool(weatherTool)).toBe(true);
  });

  test("returns true for minimal FunctionTool (no parameters)", () => {
    expect(isFunctionTool(noParamsTool)).toBe(true);
  });

  test("returns false for null/undefined", () => {
    expect(isFunctionTool(null)).toBe(false);
    expect(isFunctionTool(undefined)).toBe(false);
  });

  test("returns false for object missing type", () => {
    expect(isFunctionTool({ name: "foo", execute: () => "" })).toBe(false);
  });

  test("returns false for object missing execute", () => {
    expect(isFunctionTool({ type: "function", name: "foo" })).toBe(false);
  });

  test("returns false for hosted tool object", () => {
    expect(
      isFunctionTool({ type: "genie_space", genie_space: { id: "123" } }),
    ).toBe(false);
  });

  test("returns false for LangChain StructuredTool-like object", () => {
    const lcTool = {
      name: "lc_tool",
      description: "a langchain tool",
      invoke: async () => "result",
    };
    expect(isFunctionTool(lcTool)).toBe(false);
  });

  test("returns false for non-objects", () => {
    expect(isFunctionTool("function")).toBe(false);
    expect(isFunctionTool(42)).toBe(false);
    expect(isFunctionTool(true)).toBe(false);
  });
});

describe("functionToolToStructuredTool", () => {
  test("converts FunctionTool to StructuredToolInterface", async () => {
    const converted = functionToolToStructuredTool(weatherTool);

    expect(converted.name).toBe("get_weather");
    expect(converted.description).toBe(
      "Get the current weather for a location",
    );

    const result = await converted.invoke({ location: "Paris" });
    expect(result).toBe("Sunny in Paris");
  });

  test("handles tool with no parameters", async () => {
    const converted = functionToolToStructuredTool(noParamsTool);

    expect(converted.name).toBe("get_time");
    const result = await converted.invoke({});
    expect(result).toBeTruthy();
  });

  test("handles tool with null parameters", async () => {
    const tool: FunctionTool = {
      type: "function",
      name: "ping",
      parameters: null,
      execute: async () => "pong",
    };
    const converted = functionToolToStructuredTool(tool);
    const result = await converted.invoke({});
    expect(result).toBe("pong");
  });

  test("defaults description to empty string when not provided", async () => {
    const tool: FunctionTool = {
      type: "function",
      name: "noop",
      execute: async () => "done",
    };
    const converted = functionToolToStructuredTool(tool);
    expect(converted.description).toBe("");
  });

  test("handles nested object parameters", async () => {
    const tool: FunctionTool = {
      type: "function",
      name: "search",
      description: "Search for items",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          options: {
            type: "object",
            properties: {
              limit: { type: "integer", description: "Max results" },
            },
          },
        },
        required: ["query"],
      },
      execute: async (args) => JSON.stringify(args),
    };

    const converted = functionToolToStructuredTool(tool);
    expect(converted.name).toBe("search");

    const result = await converted.invoke({
      query: "test",
      options: { limit: 10 },
    });
    expect(JSON.parse(result)).toEqual({
      query: "test",
      options: { limit: 10 },
    });
  });

  test("handles enum parameters", async () => {
    const tool: FunctionTool = {
      type: "function",
      name: "set_mode",
      parameters: {
        type: "object",
        properties: {
          mode: {
            type: "string",
            enum: ["fast", "slow"],
            description: "Speed mode",
          },
        },
        required: ["mode"],
      },
      execute: async ({ mode }) => `Mode: ${mode}`,
    };

    const converted = functionToolToStructuredTool(tool);
    const result = await converted.invoke({ mode: "fast" });
    expect(result).toBe("Mode: fast");
  });

  test("handles optional parameters", async () => {
    const tool: FunctionTool = {
      type: "function",
      name: "greet",
      parameters: {
        type: "object",
        properties: {
          name: { type: "string" },
          greeting: { type: "string" },
        },
        required: ["name"],
      },
      execute: async ({ name, greeting }) => `${greeting ?? "Hello"}, ${name}!`,
    };

    const converted = functionToolToStructuredTool(tool);
    const result = await converted.invoke({ name: "Alice" });
    expect(result).toBe("Hello, Alice!");
  });

  test("handles boolean parameter type", async () => {
    const tool: FunctionTool = {
      type: "function",
      name: "toggle",
      parameters: {
        type: "object",
        properties: {
          enabled: { type: "boolean" },
        },
        required: ["enabled"],
      },
      execute: async ({ enabled }) => `enabled=${enabled}`,
    };

    const converted = functionToolToStructuredTool(tool);
    const result = await converted.invoke({ enabled: true });
    expect(result).toBe("enabled=true");
  });

  test("handles array parameter type", async () => {
    const tool: FunctionTool = {
      type: "function",
      name: "batch",
      parameters: {
        type: "object",
        properties: {
          items: { type: "array" },
        },
        required: ["items"],
      },
      execute: async ({ items }) => `count=${(items as any[]).length}`,
    };

    const converted = functionToolToStructuredTool(tool);
    const result = await converted.invoke({ items: [1, 2, 3] });
    expect(result).toBe("count=3");
  });

  test("handles sync execute handler", async () => {
    const tool: FunctionTool = {
      type: "function",
      name: "sync_tool",
      execute: () => "sync result",
    };

    const converted = functionToolToStructuredTool(tool);
    const result = await converted.invoke({});
    expect(result).toBe("sync result");
  });
});

describe("functionToolsToStructuredTools", () => {
  test("converts array of FunctionTools", () => {
    const converted = functionToolsToStructuredTools([
      weatherTool,
      noParamsTool,
    ]);

    expect(converted).toHaveLength(2);
    expect(converted[0].name).toBe("get_weather");
    expect(converted[1].name).toBe("get_time");
  });

  test("handles empty array", () => {
    expect(functionToolsToStructuredTools([])).toEqual([]);
  });
});
