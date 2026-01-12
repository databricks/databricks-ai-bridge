/**
 * Basic example demonstrating ChatDatabricks usage
 *
 * Set environment variables before running:
 *   DATABRICKS_HOST - Your workspace URL (e.g., https://your-workspace.databricks.com)
 *   DATABRICKS_TOKEN - Your personal access token
 *
 * Run with:
 *   npm run example
 */

import "dotenv/config";
import { ChatDatabricks } from "../src/index.js";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// Load .env.local if it exists
import { config } from "dotenv";
config({ path: ".env.local" });

async function main() {
  console.log("=== FMAPI (Foundation Model API) ===\n");

  // Initialize the model
  const model = new ChatDatabricks({
    endpoint: "databricks-meta-llama-3-3-70b-instruct",
    maxTokens: 256,
    auth: {
      host: process.env.DATABRICKS_HOST,
      token: process.env.DATABRICKS_TOKEN,
    },
  });

  // Simple string input
  const response1 = await model.invoke("What is the capital of France?");
  console.log("Q: What is the capital of France?");
  console.log(`A: ${response1.content}\n`);

  console.log("=== With System Message ===\n");

  // Using message objects
  const response2 = await model.invoke([
    new SystemMessage("You are a helpful assistant that responds concisely."),
    new HumanMessage("Explain what TypeScript is in one sentence."),
  ]);
  console.log("Q: Explain what TypeScript is in one sentence.");
  console.log(`A: ${response2.content}\n`);

  console.log("=== Streaming ===\n");

  // Streaming response
  console.log("Q: Count from 1 to 5");
  process.stdout.write("A: ");

  const stream = await model.stream("Count from 1 to 5, one number per line");
  for await (const chunk of stream) {
    process.stdout.write(chunk.content as string);
  }
  console.log("\n");

  console.log("=== Tool Calling ===\n");

  // Bind tools to the model
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

  const response3 = await modelWithTools.invoke("What's the weather like in Tokyo?");
  console.log("Q: What's the weather like in Tokyo?");

  if (response3.tool_calls && response3.tool_calls.length > 0) {
    console.log("Tool calls:");
    for (const toolCall of response3.tool_calls) {
      console.log(`  - ${toolCall.name}(${JSON.stringify(toolCall.args)})`);
    }
  } else {
    console.log(`A: ${response3.content}`);
  }

  console.log("\nDone!");
}

main().catch(console.error);
