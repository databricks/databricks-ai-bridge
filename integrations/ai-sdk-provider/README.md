# @databricks/ai-sdk-provider

Databricks provider for the [Vercel AI SDK](https://sdk.vercel.ai/docs).

## Features

- 🚀 Support for three Databricks endpoint types:
  - **Chat Agent** (`agent/v2/chat`) - Databricks chat agent API
  - **Responses Agent** (`agent/v1/responses`) - Databricks responses agent API
  - **FM API** (`llm/v1/chat`) - Foundation model chat completions API
- 🔄 Stream and non-stream (generate) support for all endpoint types
- 🛠️ Custom tool calling mechanism for Databricks agents
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

### Tool Constants

```typescript
import { DATABRICKS_TOOL_DEFINITION, DATABRICKS_TOOL_CALL_ID } from '@databricks/ai-sdk-provider'
```

#### Why are these needed?

When using AI SDK functions like `streamText` or `generateText`, you must declare which tools are allowed upfront in the `tools` parameter. This works well when you control which tools are available. However, when working with Databricks agents (like Responses agents or Agents on Apps), the agent decides which tools to call at runtime - you don't know ahead of time what tools will be invoked.

To bridge this gap, this provider uses a special "catch-all" tool pattern:

- **`DATABRICKS_TOOL_DEFINITION`**: A universal tool definition that accepts any input/output schema. This allows the provider to handle any tool that Databricks agents orchestrate, regardless of its actual schema.

- **`DATABRICKS_TOOL_CALL_ID`**: The constant ID (`'databricks-tool-call'`) used to label all tool calls and tool results under a single identifier. The actual tool name from Databricks is preserved in `providerMetadata.databricks.toolName` so it can be displayed correctly in the UI and passed back to the model.

This pattern enables dynamic tool orchestration by Databricks while maintaining compatibility with the AI SDK's tool interface.

#### Example: Server-side streaming with tools

```typescript
import { streamText } from 'ai'
import { createDatabricksProvider, DATABRICKS_TOOL_CALL_ID, DATABRICKS_TOOL_DEFINITION } from '@databricks/ai-sdk-provider'

const provider = createDatabricksProvider({
  baseURL: 'https://your-workspace.databricks.com/serving-endpoints',
  headers: { Authorization: `Bearer ${token}` },
})

const model = provider.responses('my-agent-endpoint')

const result = streamText({
  model,
  messages: convertToModelMessages(uiMessages),
  tools: {
    // Register the catch-all tool to handle any tool the agent calls
    [DATABRICKS_TOOL_CALL_ID]: DATABRICKS_TOOL_DEFINITION,
  },
})
```

When the agent makes a tool call, you'll receive it with:
- `toolName: 'databricks-tool-call'` (the constant ID)
- `providerMetadata.databricks.toolName: 'actual_tool_name'` (the real tool name)

This allows your UI to display the actual tool name while the AI SDK routes all tool calls through the single registered tool definition.

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

When your Databricks agent can call tools, register the catch-all tool definition:

```typescript
import { DATABRICKS_TOOL_CALL_ID, DATABRICKS_TOOL_DEFINITION } from '@databricks/ai-sdk-provider'

const model = provider.responses('my-agent-with-tools')

const result = await generateText({
  model,
  prompt: 'Search for information about AI',
  tools: {
    [DATABRICKS_TOOL_CALL_ID]: DATABRICKS_TOOL_DEFINITION,
  },
})

// Access tool calls from the result
for (const part of result.content) {
  if (part.type === 'tool-call') {
    // part.toolName === 'databricks-tool-call' (the constant ID)
    // part.providerMetadata.databricks.toolName contains the actual tool name
    const actualToolName = part.providerMetadata?.databricks?.toolName
    console.log(`Agent called tool: ${actualToolName}`)
  }
}
```

## Links

- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs)
- [Databricks Documentation](https://docs.databricks.com/)
- [GitHub Repository](https://github.com/databricks/databricks-ai-bridge)

## Contributing

This package is part of the [databricks-ai-bridge](https://github.com/databricks/databricks-ai-bridge) monorepo.
