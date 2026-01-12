/**
 * Advanced tool usage example demonstrating all supported BindToolsInput types
 *
 * This example shows how to use different tool definition formats with ChatDatabricks:
 * - OpenAI format (ToolDefinition)
 * - DynamicStructuredTool with Zod schemas
 * - Plain object definitions
 * - Tool execution loop (agentic pattern)
 *
 * Run with:
 *   npx tsx examples/tools.ts
 */

import "dotenv/config";
import { config } from "dotenv";
config({ path: ".env.local" });

import { z } from "zod";
import { ChatDatabricks } from "../src/index.js";
import {
  HumanMessage,
  AIMessageChunk,
  ToolMessage,
  SystemMessage,
  BaseMessage,
} from "@langchain/core/messages";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { Config } from "@databricks/sdk-experimental";

// ============================================================================
// Tool Definitions - Multiple formats supported
// ============================================================================

/**
 * 1. OpenAI Format (ToolDefinition)
 * The standard OpenAI tool format with type: "function"
 */
const weatherToolOpenAI = {
  type: "function" as const,
  function: {
    name: "get_weather",
    description: "Get the current weather for a location",
    parameters: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "The city name, e.g., 'Tokyo' or 'San Francisco'",
        },
        unit: {
          type: "string",
          enum: ["celsius", "fahrenheit"],
          description: "Temperature unit (default: celsius)",
        },
      },
      required: ["location"],
    },
  },
};

/**
 * 2. DynamicStructuredTool with Zod Schema
 * LangChain's structured tool with type-safe Zod validation
 */
const calculatorTool = new DynamicStructuredTool({
  name: "calculator",
  description: "Perform mathematical calculations. Use this for any math operations.",
  schema: z.object({
    expression: z
      .string()
      .describe("The mathematical expression to evaluate, e.g., '2 + 2' or '(10 * 5) / 2'"),
  }),
  func: async ({ expression }) => {
    try {
      // Simple eval for demo - in production use a proper math parser
      const result = Function(`"use strict"; return (${expression})`)();
      return `${expression} = ${result}`;
    } catch {
      return `Error evaluating expression: ${expression}`;
    }
  },
});

/**
 * 3. Plain Object Definition
 * Simple object with name, description, and parameters
 */
const searchTool = {
  name: "search",
  description: "Search for information on a topic. Returns relevant results.",
  parameters: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "The search query",
      },
      limit: {
        type: "number",
        description: "Maximum number of results to return (default: 5)",
      },
    },
    required: ["query"],
  },
};

/**
 * 4. Plain Object with Schema (StructuredToolParams-like)
 * Object using 'schema' instead of 'parameters'
 */
const dateTimeTool = {
  name: "get_datetime",
  description: "Get the current date and time in a specific timezone",
  schema: {
    type: "object",
    properties: {
      timezone: {
        type: "string",
        description: "The timezone, e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'",
      },
    },
    required: ["timezone"],
  },
};

// ============================================================================
// Mock Tool Execution Functions
// ============================================================================

function executeWeatherTool(args: { location: string; unit?: string }): string {
  const temps: Record<string, number> = {
    tokyo: 22,
    "san francisco": 18,
    london: 15,
    paris: 17,
    "new york": 20,
  };

  const location = args.location.toLowerCase();
  const temp = temps[location] ?? Math.floor(Math.random() * 20) + 10;
  const unit = args.unit ?? "celsius";
  const displayTemp = unit === "fahrenheit" ? Math.round((temp * 9) / 5 + 32) : temp;
  const unitSymbol = unit === "fahrenheit" ? "°F" : "°C";

  return JSON.stringify({
    location: args.location,
    temperature: displayTemp,
    unit: unitSymbol,
    conditions: ["sunny", "cloudy", "rainy", "partly cloudy"][Math.floor(Math.random() * 4)],
  });
}

function executeSearchTool(args: { query: string; limit?: number }): string {
  const limit = args.limit ?? 3;
  return JSON.stringify({
    query: args.query,
    results: [
      { title: `Result 1 for "${args.query}"`, snippet: "This is a sample search result..." },
      { title: `Result 2 for "${args.query}"`, snippet: "Another relevant result..." },
      { title: `Result 3 for "${args.query}"`, snippet: "More information about the topic..." },
    ].slice(0, limit),
  });
}

function executeDateTimeTool(args: { timezone: string }): string {
  try {
    const date = new Date();
    const formatted = date.toLocaleString("en-US", { timeZone: args.timezone });
    return JSON.stringify({
      timezone: args.timezone,
      datetime: formatted,
      iso: date.toISOString(),
    });
  } catch {
    return JSON.stringify({ error: `Invalid timezone: ${args.timezone}` });
  }
}

// Tool executor dispatcher
async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  switch (name) {
    case "get_weather":
      return executeWeatherTool(args as { location: string; unit?: string });
    case "calculator":
      return await calculatorTool.func(args as { expression: string });
    case "search":
      return executeSearchTool(args as { query: string; limit?: number });
    case "get_datetime":
      return executeDateTimeTool(args as { timezone: string });
    default:
      return JSON.stringify({ error: `Unknown tool: ${name}` });
  }
}

// ============================================================================
// Main Example
// ============================================================================

async function main() {
  console.log("=== Advanced Tool Usage Example ===\n");

  // Initialize the model
  const model = new ChatDatabricks({
    endpoint: "databricks-meta-llama-3-3-70b-instruct",
    endpointType: "fmapi",
    maxTokens: 1024,
    config: new Config({
      host: process.env.DATABRICKS_HOST,
      token: process.env.DATABRICKS_TOKEN,
    }),
  });

  // Bind all tools (demonstrating different formats)
  const tools = [weatherToolOpenAI, calculatorTool, searchTool, dateTimeTool];
  const modelWithTools = model.bindTools(tools);

  console.log("Tools bound to model:");
  console.log("  1. get_weather (OpenAI format)");
  console.log("  2. calculator (DynamicStructuredTool with Zod)");
  console.log("  3. search (Plain object with parameters)");
  console.log("  4. get_datetime (Plain object with schema)\n");

  // ============================================================================
  // Example 1: Single tool call
  // ============================================================================
  console.log("--- Example 1: Single Tool Call ---\n");

  const question1 = "What's the weather like in Tokyo?";
  console.log(`User: ${question1}\n`);

  const response1 = await modelWithTools.invoke(question1);

  if (response1.tool_calls && response1.tool_calls.length > 0) {
    console.log("Model wants to call tools:");
    for (const tc of response1.tool_calls) {
      console.log(`  → ${tc.name}(${JSON.stringify(tc.args)})`);
    }
    console.log();
  }

  // ============================================================================
  // Example 2: Multiple tool calls
  // ============================================================================
  console.log("--- Example 2: Multiple Tool Calls ---\n");

  const question2 = "What's 15% of 850, and what time is it in Tokyo?";
  console.log(`User: ${question2}\n`);

  const response2 = await modelWithTools.invoke(question2);

  if (response2.tool_calls && response2.tool_calls.length > 0) {
    console.log("Model wants to call tools:");
    for (const tc of response2.tool_calls) {
      console.log(`  → ${tc.name}(${JSON.stringify(tc.args)})`);
    }
    console.log();
  }

  // ============================================================================
  // Example 3: Full agentic loop with tool execution
  // ============================================================================
  console.log("--- Example 3: Full Agentic Loop ---\n");

  const agentQuestion =
    "I'm planning a trip to Paris. What's the weather there, and can you calculate how much 500 euros is if I tip 15%?";
  console.log(`User: ${agentQuestion}\n`);

  // Start the conversation
  const messages: BaseMessage[] = [
    new SystemMessage(
      "You are a helpful assistant. Use the available tools to answer questions. After receiving tool results, provide a final answer to the user."
    ),
    new HumanMessage(agentQuestion),
  ];

  // Agentic loop
  let iterations = 0;
  const maxIterations = 5;

  while (iterations < maxIterations) {
    iterations++;
    console.log(`[Iteration ${iterations}]`);

    const response = await modelWithTools.invoke(messages);

    // Check if the model wants to call tools
    if (response.tool_calls && response.tool_calls.length > 0) {
      console.log("Model requesting tools:");

      // Add the assistant's message with tool calls to history
      messages.push(response);

      // Execute each tool and add results
      for (const toolCall of response.tool_calls) {
        console.log(`  → Calling ${toolCall.name}(${JSON.stringify(toolCall.args)})`);

        // Execute the tool
        const result = await executeTool(toolCall.name, toolCall.args);
        console.log(`  ← Result: ${result}`);

        // Add the tool result to messages
        messages.push(
          new ToolMessage({
            content: result,
            tool_call_id: toolCall.id!,
            name: toolCall.name,
          })
        );
      }
      console.log();
    } else {
      // No more tool calls - model has final answer
      console.log(`\nAssistant: ${response.content}\n`);
      break;
    }
  }

  if (iterations >= maxIterations) {
    console.log("(Reached max iterations)\n");
  }

  // ============================================================================
  // Example 4: Streaming with tools
  // ============================================================================
  console.log("--- Example 4: Streaming with Tools ---\n");

  const streamQuestion = "Search for information about TypeScript";
  console.log(`User: ${streamQuestion}\n`);

  // Streaming agentic loop
  const streamMessages: BaseMessage[] = [
    new SystemMessage(
      "You are a helpful assistant. Use the available tools to answer questions. After receiving tool results, provide a final answer to the user."
    ),
    new HumanMessage(streamQuestion),
  ];

  let streamIterations = 0;
  const maxStreamIterations = 5;

  while (streamIterations < maxStreamIterations) {
    streamIterations++;
    console.log(`[Iteration ${streamIterations}] Streaming...`);

    const stream = await modelWithTools.stream(streamMessages);

    // Accumulate the full response
    let fullResponse = new AIMessageChunk({ content: "" });

    for await (const chunk of stream) {
      fullResponse = fullResponse.concat(chunk);

      // Stream text content to console
      if (chunk.content) {
        process.stdout.write(chunk.content as string);
      }
    }

    // Check if the model wants to call tools
    if (fullResponse.tool_calls && fullResponse.tool_calls.length > 0) {
      console.log("\nModel requesting tools:");

      // Add the assistant's message with tool calls to history
      streamMessages.push(fullResponse);

      // Execute each tool and add results
      for (const toolCall of fullResponse.tool_calls) {
        console.log(`  → Calling ${toolCall.name}(${JSON.stringify(toolCall.args)})`);

        // Execute the tool
        const result = await executeTool(toolCall.name, toolCall.args);
        console.log(`  ← Result: ${result}`);

        // Add the tool result to messages
        streamMessages.push(
          new ToolMessage({
            content: result,
            tool_call_id: toolCall.id!,
            name: toolCall.name,
          })
        );
      }
      console.log();
    } else {
      // No more tool calls - model has final answer
      console.log(`\nAssistant: ${fullResponse.content}\n`);
      break;
    }
  }

  if (streamIterations >= maxStreamIterations) {
    console.log("(Reached max iterations)\n");
  }

  console.log("\n\n=== Tool Usage Summary ===\n");
  console.log("ChatDatabricks.bindTools() accepts these tool formats:\n");
  console.log("1. OpenAI format (ToolDefinition):");
  console.log("   { type: 'function', function: { name, description, parameters } }\n");
  console.log("2. DynamicStructuredTool (with Zod schema):");
  console.log(
    "   new DynamicStructuredTool({ name, description, schema: z.object({...}), func })\n"
  );
  console.log("3. Plain object with parameters:");
  console.log("   { name, description, parameters: { type: 'object', properties: {...} } }\n");
  console.log("4. Plain object with schema:");
  console.log("   { name, description, schema: { type: 'object', properties: {...} } }\n");
  console.log("5. RunnableToolLike (via runnable.asTool()):");
  console.log("   myRunnable.asTool({ name, description, schema: z.object({...}) })\n");

  console.log("Done!");
}

main().catch(console.error);
