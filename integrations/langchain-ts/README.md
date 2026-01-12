# @databricks/langchain-ts

LangChain TypeScript integration for Databricks Model Serving.

Uses the Databricks AI SDK Provider internally to support multiple endpoint types:

- **FMAPI** - Foundation Model API (OpenAI-compatible)
- **ChatAgent** - Databricks agent chat completion
- **ResponsesAgent** - Rich output with reasoning, citations, function calls

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

## Endpoint Types

ChatDatabricks supports three endpoint types that determine which Databricks API protocol to use:

### FMAPI (Foundation Model API) - Default

OpenAI-compatible chat completions for Foundation Models.

```typescript
const model = new ChatDatabricks({
  endpoint: "databricks-meta-llama-3-3-70b-instruct",
  endpointType: "fmapi", // Default, can be omitted
});
```

### Chat Agent

Databricks agent chat completion for deployed agents.

```typescript
const model = new ChatDatabricks({
  endpoint: "my-chat-agent",
  endpointType: "chat-agent",
});
```

### Responses Agent

Rich output with reasoning, citations, and function calls.

```typescript
const model = new ChatDatabricks({
  endpoint: "my-responses-agent",
  endpointType: "responses-agent",
});
```

## Authentication

ChatDatabricks uses the `@databricks/sdk-experimental` library for authentication, which supports multiple authentication methods automatically:

1. **Environment variables** - `DATABRICKS_HOST` and `DATABRICKS_TOKEN`
2. **Databricks CLI config** - `~/.databrickscfg` file (via `databricks configure`)
3. **Azure CLI / Managed Identity** - For Azure Databricks
4. **Google Cloud credentials** - For GCP Databricks
5. **OAuth M2M** - Service Principal authentication

### Default Authentication (Recommended)

By default, credentials are automatically detected from environment variables or `~/.databrickscfg`:

```typescript
// Uses DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
// or reads from ~/.databrickscfg
const model = new ChatDatabricks({
  endpoint: "your-endpoint",
});
```

### Explicit Config

You can pass an explicit `Config` object for more control:

```typescript
import { Config } from "@databricks/sdk-experimental";

const config = new Config({
  host: "https://your-workspace.databricks.com",
  token: "dapi...",
});

const model = new ChatDatabricks({
  endpoint: "your-endpoint",
  config,
});
```

### Using a Databricks Profile

```typescript
import { Config } from "@databricks/sdk-experimental";

// Use a specific profile from ~/.databrickscfg
const config = new Config({
  profile: "my-profile",
});

const model = new ChatDatabricks({
  endpoint: "your-endpoint",
  config,
});
```

### OAuth M2M (Service Principal)

```typescript
import { Config } from "@databricks/sdk-experimental";

const config = new Config({
  host: "https://your-workspace.databricks.com",
  clientId: "your-client-id",
  clientSecret: "your-client-secret",
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
import { Config } from "@databricks/sdk-experimental";

const model = new ChatDatabricks({
  // Required
  endpoint: "your-endpoint-name",

  // Endpoint type (optional, defaults to "fmapi")
  endpointType: "fmapi" | "chat-agent" | "responses-agent",

  // Authentication (optional - uses env vars / ~/.databrickscfg by default)
  config: new Config({
    host: "https://workspace.databricks.com",
    token: "dapi...",
  }),

  // Model parameters
  temperature: 0.7, // 0.0 - 2.0
  maxTokens: 1000, // Maximum tokens to generate
  stop: ["\n\n"], // Stop sequences

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
const response1 = await modelWithTools.invoke([new HumanMessage("What's the weather in Boston?")]);

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

## Supported Endpoints

This package works with any Databricks Model Serving endpoint:

- Foundation Model APIs (e.g., `databricks-meta-llama-3-3-70b-instruct`)
- Custom model endpoints
- External model endpoints
- Databricks Agents (Chat and Responses API)

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
