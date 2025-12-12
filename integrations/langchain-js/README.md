# @databricks/langchain-ts

LangChain TypeScript integration for Databricks Model Serving.

## Installation

```bash
npm install @databricks/langchain-ts @langchain/core
```

## Quick Start

```typescript
import { ChatDatabricks } from "@databricks/langchain-ts";

const model = new ChatDatabricks({
  endpoint: "databricks-meta-llama-3-3-70b-instruct",
});

const response = await model.invoke("Hello, how are you?");
console.log(response.content);
```

## Authentication

The package uses the [Databricks SDK](https://www.npmjs.com/package/@databricks/sdk-experimental) for authentication. It automatically detects credentials from:

1. Environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)
2. Databricks CLI configuration (`~/.databrickscfg`)
3. Azure CLI / Managed Identity (when running on Azure)
4. Google Cloud credentials (when running on GCP)

### Explicit Configuration

```typescript
// Using host and token
const model = new ChatDatabricks({
  endpoint: "your-endpoint",
  host: "https://your-workspace.databricks.com",
  token: "dapi...",
});

// Using Config object for advanced scenarios
import { Config } from "@databricks/sdk-experimental";

const config = new Config({
  host: "https://your-workspace.databricks.com",
  // OAuth, Azure, or GCP configuration
});

const model = new ChatDatabricks({
  endpoint: "your-endpoint",
  config,
});
```

## Streaming

```typescript
const stream = await model.stream("Tell me a story");

for await (const chunk of stream) {
  process.stdout.write(chunk.content as string);
}
```

## Tool Calling

```typescript
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

const response = await modelWithTools.invoke("What's the weather in NYC?");

if (response.tool_calls) {
  for (const toolCall of response.tool_calls) {
    console.log(`Tool: ${toolCall.name}`);
    console.log(`Args: ${JSON.stringify(toolCall.args)}`);
  }
}
```

### Using LangChain Tools

```typescript
import { z } from "zod";
import { tool } from "@langchain/core/tools";

const weatherTool = tool(
  async ({ location }) => {
    return `The weather in ${location} is sunny, 72°F`;
  },
  {
    name: "get_weather",
    description: "Get weather for a location",
    schema: z.object({
      location: z.string().describe("City and state"),
    }),
  }
);

const modelWithTools = model.bindTools([weatherTool]);
```

## Configuration Options

```typescript
const model = new ChatDatabricks({
  // Required
  endpoint: "your-endpoint-name",

  // Authentication (optional if using environment/CLI config)
  host: "https://workspace.databricks.com",
  token: "dapi...",
  config: new Config({ /* ... */ }),

  // Model parameters
  temperature: 0.7,      // 0.0 - 2.0
  maxTokens: 1000,       // Maximum tokens to generate
  stop: ["\n\n"],        // Stop sequences
  n: 1,                  // Number of completions

  // Streaming
  streamUsage: true,     // Include usage in streaming responses

  // HTTP settings
  timeout: 60000,        // Request timeout (ms)
  maxRetries: 2,         // Retry count for transient failures

  // Extra parameters (passed directly to the model)
  extraParams: {
    top_p: 0.9,
  },
});
```

## Call-Time Options

Options can also be passed at call time:

```typescript
const response = await model.invoke("Hello", {
  temperature: 0.5,
  maxTokens: 100,
  stop: ["."],
});
```

## Multi-Turn Conversations

```typescript
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";

const response = await model.invoke([
  new SystemMessage("You are a helpful assistant."),
  new HumanMessage("What's the capital of France?"),
  new AIMessage("The capital of France is Paris."),
  new HumanMessage("What's its population?"),
]);
```

## Tool Calling with Responses

```typescript
import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";

// First call - model requests tool
const response1 = await modelWithTools.invoke([
  new HumanMessage("What's the weather in Boston?"),
]);

// Execute tool and continue conversation
if (response1.tool_calls?.length) {
  const toolResult = await executeWeatherAPI(response1.tool_calls[0].args);

  const response2 = await modelWithTools.invoke([
    new HumanMessage("What's the weather in Boston?"),
    response1,
    new ToolMessage({
      content: JSON.stringify(toolResult),
      tool_call_id: response1.tool_calls[0].id!,
    }),
  ]);

  console.log(response2.content);
}
```

## Error Handling

```typescript
import { DatabricksRequestError } from "@databricks/langchain-ts";

try {
  const response = await model.invoke("Hello");
} catch (error) {
  if (error instanceof DatabricksRequestError) {
    console.error(`Status: ${error.status}`);
    console.error(`Error code: ${error.errorCode}`);
    console.error(`Message: ${error.message}`);
  }
}
```

## Supported Endpoints

This package works with any Databricks Model Serving endpoint that implements the OpenAI-compatible chat completions API:

- Foundation Model APIs (e.g., `databricks-meta-llama-3-3-70b-instruct`)
- Custom model endpoints
- External model endpoints

## Examples

See the [examples](./examples) folder for complete working examples.

```bash
# Copy the example env file and fill in your credentials
cp .env.example .env.local

# Edit .env.local with your Databricks host and token
# Then run the example
npm run example
```

Alternatively, set environment variables directly:

```bash
export DATABRICKS_HOST=https://your-workspace.databricks.com
export DATABRICKS_TOKEN=dapi...
npm run example
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run unit tests
npm test

# Run integration tests (requires Databricks credentials)
export DATABRICKS_HOST=https://your-workspace.databricks.com
export DATABRICKS_TOKEN=dapi...
export TEST_ENDPOINT_NAME=databricks-meta-llama-3-3-70b-instruct
npm run test:integration

# Lint
npm run lint

# Format
npm run format
```

## License

Apache-2.0
