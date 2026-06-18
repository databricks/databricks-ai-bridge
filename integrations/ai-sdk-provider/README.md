# @databricks/ai-sdk-provider

Databricks provider for the [Vercel AI SDK](https://sdk.vercel.ai/docs).

This package supports two integration styles:

- **Language model provider** for Databricks serving endpoints
- **Genie conversation client and custom agent** for Databricks Genie conversation APIs

## Features

- 🚀 Support for all Databricks endpoint types:
  - **Responses** (`agent/v1/responses`) - Foundation model and agent responses API ([docs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#responses-api))
  - **Chat Completions** (`llm/v1/chat`) - Foundation model chat completions API ([docs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#chat-completions-api))
  - **Chat Agent** (`agent/v2/chat`) - Legacy Databricks chat agent API ([docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-legacy-schema))
- 🔄 Stream and non-stream (generate) support for all endpoint types
- 🛠️ Tool calling and agent support
- 💬 Genie conversation helper and custom AI SDK agent support
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

To use the custom Genie agent with AI SDK agent utilities, also install:

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

## Genie Quick Start

```typescript
import { createAgentUIStreamResponse } from 'ai'
import { createDatabricksGenieAgent } from '@databricks/ai-sdk-provider'

const genieAgent = createDatabricksGenieAgent({
  baseURL: 'https://your-workspace.databricks.com',
  spaceId: 'your-genie-space-id',
  headers: {
    Authorization: `Bearer ${token}`,
  },
})

export async function POST(request: Request) {
  const { messages } = await request.json()

  return createAgentUIStreamResponse({
    agent: genieAgent,
    uiMessages: messages,
  })
}
```

The returned AI SDK result keeps Genie-specific metadata on `result.genie`, including:

- `conversationId`
- `messageId`
- `status`
- `text`
- `sql`
- `attachmentId`
- `suggestedQuestions`
- optional `queryResult`

## Genie Live Smoke Test

You can run a real end-to-end smoke test against your Databricks workspace with:

```bash
npm run test:genie:live
```

The test auto-loads `.env.local` from this package directory before execution. Set:

```bash
DATABRICKS_HOST="https://your-workspace.databricks.com"
DATABRICKS_TOKEN="your-token"
DATABRICKS_GENIE_SPACE_ID="your-genie-space-id"
```

Optional environment variables:

- `DATABRICKS_GENIE_TEST_QUESTION`: override the first Genie question
- `DATABRICKS_GENIE_TEST_FOLLOW_UP_QUESTION`: override the follow-up question sent through the agent

The live smoke test:

- auto-loads `.env.local`
- starts with the Genie conversation client
- logs raw and normalized attachment details
- uses Genie-suggested follow-up questions when they are available
- fetches tabular query results when Genie returns a query attachment
- sends a second follow-up through the Genie agent using the returned `conversationId`

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

The Genie conversation client and Genie agent use the same provider-style settings, but the `baseURL`
should be your workspace host rather than `/serving-endpoints`.

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

### Genie Exports

#### `createDatabricksGenieConversationClient(settings)`

Creates a conversation-focused client for the Databricks Genie conversation API.

**Parameters:**

- `settings.baseURL` (string, required): Databricks workspace base URL
- `settings.spaceId` (string, required): Genie space ID
- `settings.headers` (object, optional): Custom headers to include in requests
- `settings.fetch` (function, optional): Custom fetch implementation
- `settings.formatUrl` (function, optional): Optional function to format the URL
- `settings.timeoutMs` (number, optional): Polling timeout, default `600000`
- `settings.initialPollIntervalMs` (number, optional): Initial polling delay, default `1000`
- `settings.maxPollIntervalMs` (number, optional): Maximum polling delay, default `60000`
- `settings.backoffMultiplier` (number, optional): Polling backoff multiplier, default `2`

The conversation client exposes:

- `startConversation(question)`
- `createMessage(conversationId, question)`
- `getMessage(conversationId, messageId)`
- `waitForCompletion(conversationId, messageId, options)`
- `getQueryResult(conversationId, messageId, attachmentId)`
- `ask(question, options)`

`ask(question, options)` returns a normalized result with:

- `conversationId`
- `messageId`
- `status`
- `text`
- `sql`
- `attachmentId`
- `suggestedQuestions`
- optional `queryResult`
- the normalized raw `message`

#### `createDatabricksGenieAgent(settings)`

Creates a custom AI SDK agent backed directly by the Databricks Genie conversation API.

The agent:

- sends only the latest user message as the new Genie question
- keeps Databricks conversation state via `conversationId`
- returns normal assistant text plus structured Genie metadata in `result.genie`
- supports `createAgentUIStreamResponse()` without requiring a new language-model provider

Agent call options:

- `conversationId?: string`
- `fetchQueryResult?: boolean` default `false`
- `headers?: Record<string, string>`
- polling overrides: `timeoutMs`, `initialPollIntervalMs`, `maxPollIntervalMs`, `backoffMultiplier`

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
