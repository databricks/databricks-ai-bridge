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

**Returns:** `DatabricksProvider` with three model creation methods:

- `responses(modelId: string)`: Create a Responses model
- `chatCompletions(modelId: string)`: Create a Chat Completions model
- `chatAgent(modelId: string)`: Create a Chat Agent model

### Dynamic Tool Calling

When working with Databricks agents (like Responses agents or Agents on Apps), the agent decides which tools to call at runtime - you don't know ahead of time what tools will be invoked.

This provider handles this automatically by marking all tool calls as `dynamic: true`. This means:

- **No pre-registration required**: You don't need to declare tools upfront in the `tools` parameter
- **Actual tool names**: Tool calls use the real tool name from Databricks, not a placeholder
- **Provider-executed**: The AI SDK knows these tools are handled remotely by Databricks

#### Example: Server-side streaming with tools

```typescript
import { streamText } from 'ai'
import { createDatabricksProvider } from '@databricks/ai-sdk-provider'

const provider = createDatabricksProvider({
  baseURL: 'https://your-workspace.databricks.com/serving-endpoints',
  headers: { Authorization: `Bearer ${token}` },
})

const model = provider.responses('my-agent-endpoint')

const result = streamText({
  model,
  messages: convertToModelMessages(uiMessages),
  // No need to pre-register tools - they're handled dynamically
})
```

When the agent makes a tool call, you'll receive it with the actual tool name directly in `toolName`.

### MCP Utilities

```typescript
import {
  MCP_APPROVAL_STATUS_KEY,
  MCP_APPROVAL_REQUEST_TYPE,
  MCP_APPROVAL_RESPONSE_TYPE,
  isMcpApprovalRequest,
  isMcpApprovalResponse,
  createApprovalStatusOutput,
  getMcpApprovalState,
} from '@databricks/ai-sdk-provider'
```

MCP (Model Context Protocol) approval utilities for handling approval workflows.

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

### With Tool Calling

When your Databricks agent calls tools, the tool calls are handled automatically:

```typescript
const model = provider.responses('my-agent-with-tools')

const result = await generateText({
  model,
  prompt: 'Search for information about AI',
  // No need to pre-register tools - they're handled dynamically
})

// Access tool calls from the result
for (const part of result.content) {
  if (part.type === 'tool-call') {
    // part.toolName contains the actual tool name
    console.log(`Agent called tool: ${part.toolName}`)
  }
}
```

## Links

- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs)
- [Databricks Documentation](https://docs.databricks.com/)
- [GitHub Repository](https://github.com/databricks/databricks-ai-bridge)

## Contributing

This package is part of the [databricks-ai-bridge](https://github.com/databricks/databricks-ai-bridge) monorepo.
