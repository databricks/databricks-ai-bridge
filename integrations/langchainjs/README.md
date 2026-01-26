# @databricks/langchainjs

LangChain TypeScript integration for Databricks Model Serving.

This package provides a `ChatDatabricks` class that integrates with the LangChain ecosystem, allowing you to use Databricks Model Serving endpoints with LangChain's chat model interface.

## Features

- Compatible with LangChain's `BaseChatModel` interface
- Supports streaming responses
- Supports tool/function calling
- Multiple endpoint APIs: Chat Completions, Chat Agent, and Responses
- Automatic authentication via Databricks SDK

## Requirements

- Node.js >= 18.0.0
- A Databricks workspace with Model Serving enabled

## Installation

```bash
npm install @databricks/langchainjs
```

## Quick Start

```typescript
import { ChatDatabricks } from "@databricks/langchainjs";

const model = new ChatDatabricks({
  endpoint: "databricks-meta-llama-3-3-70b-instruct",
  endpointAPI: "chat-completions",
});

const response = await model.invoke("Hello, how are you?");
console.log(response.content);
```

## Endpoint APIs

ChatDatabricks supports three endpoint APIs via the `endpointAPI` parameter:

### Chat Completions

OpenAI-compatible chat completions for Foundation Models.

```typescript
const model = new ChatDatabricks({
  endpoint: "databricks-meta-llama-3-3-70b-instruct",
  endpointAPI: "chat-completions",
});
```

### Chat Agent

Databricks agent chat completion for deployed agents.

```typescript
const model = new ChatDatabricks({
  endpoint: "my-chat-agent",
  endpointAPI: "chat-agent",
});
```

### Responses

Rich output with reasoning, citations, and function calls.

```typescript
const model = new ChatDatabricks({
  endpoint: "my-responses-agent",
  endpointAPI: "responses",
});
```

## Authentication

ChatDatabricks uses the [Databricks SDK](https://github.com/databricks/databricks-sdk-js?tab=readme-ov-file#authentication) for authentication, which automatically detects credentials from:

- Environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)
- Databricks CLI config (`~/.databrickscfg`)
- Azure CLI / Managed Identity
- Google Cloud credentials
- OAuth M2M (Service Principal)

```typescript
// Credentials are automatically detected
const model = new ChatDatabricks({
  endpoint: "your-endpoint",
  endpointAPI: "chat-completions",
});
```

### Explicit Auth

You can also pass credentials directly via the `auth` field:

```typescript
const model = new ChatDatabricks({
  endpoint: "your-endpoint",
  endpointAPI: "chat-completions",
  auth: {
    host: "https://your-workspace.databricks.com",
    token: "dapi...",
  },
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

## Using with LangChain Agents

`ChatDatabricks` works with LangChain's `createAgent`:

```typescript
import { createAgent } from "langchain";
import { ChatDatabricks } from "@databricks/langchainjs";

const model = new ChatDatabricks({
  endpoint: "databricks-meta-llama-3-3-70b-instruct",
  endpointAPI: "chat-completions",
});

const agent = createAgent({
  llm: model,
  tools: [weatherTool, searchTool],
});

const result = await agent.invoke("What's the weather in Paris?");
```

## Configuration Options

```typescript
const model = new ChatDatabricks({
  // Required
  endpoint: "your-endpoint-name",
  endpointAPI: "chat-completions", // or "chat-agent" or "responses"

  // Model parameters (optional)
  temperature: 0.7,
  maxTokens: 1000,
  stop: ["\n\n"],
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

# Run the basic example
npm run example

# Run the tools example
npm run example:tools
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
npm run test:integration

# Type check
npm run typecheck

# Lint and format
npm run lint
npm run format
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## Links

- [Databricks AI Bridge](https://github.com/databricks/databricks-ai-bridge)
- [LangChain.js](https://js.langchain.com/)
- [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [Databricks TypeScript SDK](https://docs.databricks.com/en/dev-tools/sdk-typescript.html)
