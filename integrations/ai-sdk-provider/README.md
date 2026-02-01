# @databricks/ai-sdk-provider

Databricks provider for the [Vercel AI SDK](https://sdk.vercel.ai/docs).

## Features

- 🚀 Support for all Databricks endpoint types:
  - **Responses** (`agent/v1/responses`) - Foundation model and agent responses API ([docs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#responses-api))
  - **Chat Completions** (`llm/v1/chat`) - Foundation model chat completions API ([docs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#chat-completions-api))
  - **Chat Agent** (`agent/v2/chat`) - Legacy Databricks chat agent API ([docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-legacy-schema))
- 🔄 Stream and non-stream (generate) support for all endpoint types
- 🛠️ Tool calling and agent support
- 🔐 Flexible authentication (bring your own tokens/headers)
- 🎯 Full TypeScript support

## Installation

```bash
npm install @databricks/ai-sdk-provider
```

## Peer Dependencies

This package requires the following peer dependencies:

```bash
npm install @ai-sdk/provider @ai-sdk/provider-utils
```

To use the provider with AI SDK functions like `generateText` or `streamText`, also install:

```bash
npm install ai
```

## Quick Start

```typescript
import { createDatabricksProvider } from '@databricks/ai-sdk-provider'
import { generateText } from 'ai'

// Create provider with your workspace URL and authentication
const provider = createDatabricksProvider({
  baseURL: 'https://your-workspace.databricks.com/serving-endpoints',
  headers: {
    Authorization: `Bearer ${token}`,
  },
})

// Use the Responses endpoint
const model = provider.responses('your-agent-endpoint')

const result = await generateText({
  model,
  prompt: 'Hello, how are you?',
})

console.log(result.text)
```

## Authentication

The provider requires you to pass authentication headers:

```typescript
const provider = createDatabricksProvider({
  baseURL: 'https://your-workspace.databricks.com/serving-endpoints',
  headers: {
    Authorization: `Bearer ${token}`,
  },
})
```

## API Reference

### Main Export

#### `createDatabricksProvider(settings)`

Creates a Databricks provider instance.

**Parameters:**

- `settings.baseURL` (string, required): Base URL for the Databricks API calls
- `settings.headers` (object, optional): Custom headers to include in requests
- `settings.provider` (string, optional): Provider name (defaults to "databricks")
- `settings.fetch` (function, optional): Custom fetch implementation
- `settings.formatUrl` (function, optional): Optional function to format the URL
- `settings.useRemoteToolCalling` (boolean, optional): Enable remote tool calling mode (defaults to `false`). See [Remote Tool Calling](#remote-tool-calling) below.

**Returns:** `DatabricksProvider` with three model creation methods:

- `responses(modelId: string)`: Create a Responses model
- `chatCompletions(modelId: string)`: Create a Chat Completions model
- `chatAgent(modelId: string)`: Create a Chat Agent model

### Remote Tool Calling

The `useRemoteToolCalling` option controls how tool calls from Databricks agents are handled. When enabled, tool calls are marked as `dynamic: true` and `providerExecuted: true`, which tells the AI SDK that:

1. **Dynamic**: The tools are not pre-registered - the agent decides which tools to call at runtime
2. **Provider-executed**: The tools are executed remotely by Databricks, not by your application

#### When to use `useRemoteToolCalling: true`

Enable this option when your Databricks agent handles tool execution internally:

- **Databricks Agents with built-in tools**: Agents that use tools like Python execution, SQL queries, or other Databricks-managed tools
- **Agents on Apps**: When deploying agents that manage their own tool execution
- **MCP (Model Context Protocol) integrations**: When tools are executed via MCP servers managed by Databricks

```typescript
const provider = createDatabricksProvider({
  baseURL: 'https://your-workspace.databricks.com/serving-endpoints',
  headers: { Authorization: `Bearer ${token}` },
  useRemoteToolCalling: true, // Enable for Databricks-managed tool execution
})
```

#### When NOT to use `useRemoteToolCalling`

Keep this option disabled (the default) when:

- **You define and execute tools locally**: Your application registers tools with the AI SDK and handles their execution
- **Standard chat completions**: You're using the Chat Completions endpoint without agent features
- **Hybrid scenarios**: You want to intercept tool calls and handle some locally

```typescript
// Default behavior - you handle tool execution
const provider = createDatabricksProvider({
  baseURL: 'https://your-workspace.databricks.com/serving-endpoints',
  headers: { Authorization: `Bearer ${token}` },
  // useRemoteToolCalling defaults to false
})

const result = await generateText({
  model: provider.chatCompletions('my-model'),
  prompt: 'What is the weather?',
  tools: {
    getWeather: {
      description: 'Get weather for a location',
      parameters: z.object({ location: z.string() }),
      execute: async ({ location }) => {
        // Your local tool execution
        return fetchWeather(location)
      },
    },
  },
})
```

#### Example: Remote tool calling with Databricks agents

```typescript
import { streamText } from 'ai'
import { createDatabricksProvider } from '@databricks/ai-sdk-provider'

const provider = createDatabricksProvider({
  baseURL: 'https://your-workspace.databricks.com/serving-endpoints',
  headers: { Authorization: `Bearer ${token}` },
  useRemoteToolCalling: true,
})

const model = provider.responses('my-agent-endpoint')

const result = streamText({
  model,
  messages: convertToModelMessages(uiMessages),
  // No need to pre-register tools - they're handled by Databricks
})

// Tool calls will have the actual tool name from Databricks
for await (const part of result.fullStream) {
  if (part.type === 'tool-call') {
    console.log(`Agent called: ${part.toolName}`)
    // Tool is executed remotely - result will come from Databricks
  }
}
```

## Examples

### Responses Endpoint

```typescript
const model = provider.responses('my-responses-agent')

const result = await generateText({
  model,
  prompt: 'Analyze this data...',
})

console.log(result.text)
```

## Links

- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs)
- [Databricks Documentation](https://docs.databricks.com/)
- [GitHub Repository](https://github.com/databricks/databricks-ai-bridge)

## Contributing

This package is part of the [databricks-ai-bridge](https://github.com/databricks/databricks-ai-bridge) monorepo.
