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

// Use the Chat Agent endpoint
const responsesAgent = provider.responsesAgent('your-agent-endpoint')

const result = await generateText({
  model: responsesAgent,
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

- `chatAgent(modelId: string)`: Create a Chat Agent model
- `responsesAgent(modelId: string)`: Create a Responses Agent model
- `fmapi(modelId: string)`: Create an FM API model

### Tool Constants

```typescript
import { DATABRICKS_TOOL_DEFINITION, DATABRICKS_TOOL_CALL_ID } from '@databricks/ai-sdk-provider'
```

#### Why are these needed?

The AI SDK requires tools to be defined ahead of time with known schemas. However, Databricks agents can orchestrate tools dynamically at runtime - we don't know which tools will be called until the model executes.

To bridge this gap, this provider uses a special "catch-all" tool definition:

- **`DATABRICKS_TOOL_DEFINITION`**: A universal tool definition that accepts any input/output schema. This allows the provider to handle any tool that Databricks agents orchestrate, regardless of its schema.

- **`DATABRICKS_TOOL_CALL_ID`**: The constant ID (`'databricks-tool-call'`) used to identify this special tool. The actual tool name from Databricks is preserved in the metadata so it can be displayed correctly in the UI and passed back to the model.

This pattern enables dynamic tool orchestration by Databricks while maintaining compatibility with the AI SDK's tool interface.

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

### Responses Agent Endpoint

```typescript
const responsesAgent = provider.responsesAgent('my-responses-agent')

const result = await generateText({
  model: responsesAgent,
  prompt: 'Analyze this data...',
})

console.log(result.text)
```

### With Tool Calling

```typescript
import { DATABRICKS_TOOL_CALL_ID, DATABRICKS_TOOL_DEFINITION } from '@databricks/ai-sdk-provider'

const responsesAgent = provider.responsesAgent('my-agent-with-tools')

const result = await generateText({
  model: responsesAgent,
  prompt: 'Search for information about AI',
  tools: {
    [DATABRICKS_TOOL_CALL_ID]: DATABRICKS_TOOL_DEFINITION,
  },
})
```

## Links

- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs)
- [Databricks Documentation](https://docs.databricks.com/)
- [GitHub Repository](https://github.com/databricks/databricks-ai-bridge)

## Contributing

This package is part of the [databricks-ai-bridge](https://github.com/databricks/databricks-ai-bridge) monorepo.
