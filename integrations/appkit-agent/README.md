# @databricks/appkit-agent

Agent plugin for [Databricks AppKit](https://github.com/databricks/appkit). Provides two things:

1. **`AgentInterface`** — a contract for writing custom agent implementations that speak the OpenAI Responses API format (streaming + non-streaming).
2. **`StandardAgent`** — a ready-to-use LangGraph-based ReAct agent that implements `AgentInterface`, with streaming Responses API support, function tools, and Databricks-hosted tool integration (Genie, Vector Search, MCP servers).

## Installation

```bash
npm install @databricks/appkit-agent
```

The LangChain peer dependencies are required when using the built-in ReAct agent (not needed if you provide a custom `agentInstance`):

```bash
npm install @databricks/langchainjs @langchain/core @langchain/langgraph
```

If you use hosted MCP tools (Genie, Vector Search, custom/external MCP servers):

```bash
npm install @langchain/mcp-adapters
```

## Quick Start

```typescript
import { createApp, server } from "@databricks/appkit";
import { agent } from "@databricks/appkit-agent";

const app = await createApp({
  plugins: [
    server(),
    agent({
      model: "databricks-claude-sonnet-4-5",
      systemPrompt: "You are a helpful assistant.",
    }),
  ],
});

app.server.start();
```

The plugin registers `POST /api/agent` which accepts the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) request format with SSE streaming.

## Environment Variables

| Variable           | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `DATABRICKS_MODEL` | Default model serving endpoint name. Overridden by `config.model`. |

## Configuration

```typescript
agent({
  // Model serving endpoint (or set DATABRICKS_MODEL env var)
  model: "databricks-claude-sonnet-4-5",

  // System prompt injected at the start of every conversation
  systemPrompt: "You are a helpful assistant.",

  // Tools available to the agent (see Tools section below)
  tools: [myTool, genieTool],

  // Or bring your own AgentInterface implementation (skips LangGraph setup)
  agentInstance: myCustomAgent,
});
```

## Tools

The agent supports two kinds of tools: **function tools** (local code) and **hosted tools** (Databricks-managed services).

### Function Tools

Define tools as plain objects following the OpenResponses FunctionTool schema:

```typescript
import type { FunctionTool } from "@databricks/appkit-agent";

const weatherTool: FunctionTool = {
  type: "function",
  name: "get_weather",
  description: "Get the current weather for a location",
  parameters: {
    type: "object",
    properties: {
      location: {
        type: "string",
        description: "City name, e.g. 'San Francisco'",
      },
    },
    required: ["location"],
  },
  execute: async ({ location }) => {
    // Call your weather API here
    return `Weather in ${location}: sunny, 72°F`;
  },
};

agent({ model: "databricks-claude-sonnet-4-5", tools: [weatherTool] });
```

### Hosted Tools

Connect to Databricks-managed services without writing tool handlers:

```typescript
// Genie Space — natural-language queries over your data
const genie = {
  type: "genie-space" as const,
  genie_space: { id: "01efg..." },
};

// Vector Search Index — semantic search over indexed documents
const vectorSearch = {
  type: "vector_search_index" as const,
  vector_search_index: { name: "catalog.schema.my_index" },
};

// Custom MCP Server — a Databricks App running an MCP server
const customMcp = {
  type: "custom_mcp_server" as const,
  custom_mcp_server: { app_name: "my-app", app_url: "my-app-url" },
};

// External MCP Server — a Unity Catalog connection to an external MCP endpoint
const externalMcp = {
  type: "external_mcp_server" as const,
  external_mcp_server: { connection_name: "my-connection" },
};

agent({
  model: "databricks-claude-sonnet-4-5",
  tools: [genie, vectorSearch, customMcp, externalMcp],
});
```

### Adding Tools After Creation

```typescript
const app = await createApp({
  plugins: [
    server(),
    agent({ model: "databricks-claude-sonnet-4-5", tools: [weatherTool] }),
  ],
});

// Add more tools after the app is running
await app.agent.addTools([timeTool]);
```

## Programmatic API

After `createApp`, the plugin exposes methods on `app.agent`:

```typescript
// Non-streaming invoke — returns the assistant's text reply
const reply = await app.agent.invoke([
  { role: "user", content: "What's the weather in SF?" },
]);

// Streaming — yields Responses API SSE events
for await (const event of app.agent.stream([
  { role: "user", content: "Tell me a story" },
])) {
  if (event.type === "response.output_text.delta") {
    process.stdout.write(event.delta);
  }
}

// Add tools dynamically
await app.agent.addTools([myNewTool]);
```

## Databricks Apps Deployment

Databricks product UIs (AI Playground, Agent Evaluation, the built-in chat UI) interact with agents via the `/invocations` endpoint by convention. Since the AppKit agent plugin mounts at `/api/agent` by default, add a redirect so these UIs can chat with your agent:

```typescript
app.server.extend((expressApp) => {
  expressApp.post("/invocations", (req, res) => {
    req.url = "/api/agent";
    expressApp(req, res);
  });
});

app.server.start();
```

## Custom Agent Implementation

To bring your own agent logic, implement the `AgentInterface` and pass it as `agentInstance`:

```typescript
import type {
  AgentInterface,
  InvokeParams,
  ResponseOutputItem,
  ResponseStreamEvent,
} from "@databricks/appkit-agent";

class MyAgent implements AgentInterface {
  async invoke(params: InvokeParams): Promise<ResponseOutputItem[]> {
    // Your invoke logic — return Responses API output items
  }

  async *stream(params: InvokeParams): AsyncGenerator<ResponseStreamEvent> {
    // Your streaming logic — yield Responses API SSE events
  }
}

agent({ agentInstance: new MyAgent() });
```

The `StandardAgent` class (exported from this package) is the built-in implementation that wraps a LangGraph `createReactAgent` and translates its stream events into Responses API format. When you pass `model` instead of `agentInstance`, the plugin uses `StandardAgent` under the hood.

## API Reference

### Exports

| Export                | Kind           | Description                                              |
| --------------------- | -------------- | -------------------------------------------------------- |
| `agent`               | Plugin factory | Main entry point — call with config, pass to `createApp` |
| `StandardAgent`       | Class          | LangGraph-backed `AgentInterface` implementation         |
| `createInvokeHandler` | Function       | Express handler factory for the `/api/agent` endpoint    |
| `isFunctionTool`      | Function       | Type guard for `FunctionTool`                            |
| `isHostedTool`        | Function       | Type guard for `HostedTool`                              |

### Types

| Type                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `IAgentConfig`        | Plugin configuration options                                  |
| `AgentInterface`      | Contract for custom agent implementations                     |
| `AgentTool`           | Union of `FunctionTool \| HostedTool`                         |
| `FunctionTool`        | Local tool with JSON Schema parameters and `execute` handler  |
| `HostedTool`          | Union of Genie, Vector Search, Custom MCP, External MCP tools |
| `InvokeParams`        | Input to `invoke()` / `stream()`                              |
| `ResponseOutputItem`  | Output item (message, function call, or function call output) |
| `ResponseStreamEvent` | SSE event types for streaming responses                       |

## License

See [LICENSE](./LICENSE).
