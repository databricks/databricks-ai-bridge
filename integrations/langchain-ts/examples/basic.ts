/**
 * Basic example demonstrating ChatDatabricks usage
 *
 * Authentication is handled automatically by the Databricks SDK.
 * It will try these methods in order:
 * 1. Environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN)
 * 2. Databricks CLI config (~/.databrickscfg)
 * 3. Azure CLI / Managed Identity (for Azure Databricks)
 * 4. Google Cloud credentials (for GCP Databricks)
 *
 * You can also pass an explicit Config object for more control.
 *
 * Run with:
 *   npm run example
 */

import "dotenv/config";
import { ChatDatabricks } from "../src/index.js";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Config } from "@databricks/sdk-experimental";

// Load .env.local if it exists
import { config } from "dotenv";
config({ path: ".env.local" });

async function main() {
  console.log("=== FMAPI (Foundation Model API) ===\n");

  // Initialize the model with FMAPI endpoint (default)
  const model = new ChatDatabricks({
    endpoint: "databricks-meta-llama-3-3-70b-instruct",
    endpointType: "fmapi", // Default, can be omitted
    maxTokens: 256,
    config: new Config({
      host: process.env.DATABRICKS_HOST,
      token: process.env.DATABRICKS_TOKEN,
    }),
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

  console.log("response3", response3);

  if (response3.tool_calls && response3.tool_calls.length > 0) {
    console.log("Tool calls:");
    for (const toolCall of response3.tool_calls) {
      console.log(`  - ${toolCall.name}(${JSON.stringify(toolCall.args)})`);
    }
  } else {
    console.log(`A: ${response3.content}`);
  }

  console.log("\n=== Other Endpoint Types ===\n");
  console.log("ChatDatabricks supports three endpoint types:\n");
  console.log("1. FMAPI (Foundation Model API) - OpenAI-compatible chat completions");
  console.log('   const model = new ChatDatabricks({ endpoint: "...", endpointType: "fmapi" });\n');
  console.log("2. Chat Agent - Databricks agent chat completion");
  console.log(
    '   const model = new ChatDatabricks({ endpoint: "my-agent", endpointType: "chat-agent" });\n'
  );
  console.log("3. Responses Agent - Rich output with reasoning, citations, function calls");
  console.log(
    '   const model = new ChatDatabricks({ endpoint: "my-agent", endpointType: "responses-agent" });\n'
  );

  console.log("=== Authentication Options ===\n");
  console.log("Authentication is handled automatically by the Databricks SDK.\n");
  console.log("Default (env vars or ~/.databrickscfg):");
  console.log('  const model = new ChatDatabricks({ endpoint: "..." });\n');
  console.log("Explicit Config:");
  console.log("  import { Config } from '@databricks/sdk-experimental';");
  console.log("  const config = new Config({ host: '...', token: '...' });");
  console.log('  const model = new ChatDatabricks({ endpoint: "...", config });\n');
  console.log("Using a profile:");
  console.log("  const config = new Config({ profile: 'my-profile' });");
  console.log('  const model = new ChatDatabricks({ endpoint: "...", config });\n');

  console.log("Done!");
}

main().catch(console.error);
