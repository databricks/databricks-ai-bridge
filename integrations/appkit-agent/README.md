# @databricks/appkit-agent

Plugins for [Databricks AppKit](https://github.com/databricks/appkit):

- **Agent plugin** — define an AI agent declaratively (model + tools + instructions) or bring a custom `AgentInterface` implementation. Speaks the OpenAI Responses API format with streaming.
- **Chat plugin** — full-featured chat server with streaming, session management, optional PostgreSQL persistence, feedback via MLflow Traces, and stream resumption.

## Installation

```bash
npm install @databricks/appkit-agent
```

The following peer dependencies are required when using the built-in agent (not needed if you provide a custom `agentInstance`):

```bash
npm install @databricks/langchainjs @langchain/core @langchain/langgraph
```

If you use hosted tools (Genie, Vector Search, custom/external MCP servers):

```bash
npm install @langchain/mcp-adapters
```

For chat persistence (optional — without it the chat plugin runs in ephemeral mode):

```bash
npm install pg
```

## Quick Start

### Agent only

```typescript
import { createApp, server } from "@databricks/appkit";
import { agent } from "@databricks/appkit-agent";

const app = await createApp({
  plugins: [
    server({ autoStart: true }),
    agent({
      model: "databricks-claude-sonnet-4-5",
      systemPrompt: "You are a helpful assistant.",
    }),
  ],
});
```

### Agent + Chat

```typescript
import { createApp, server } from "@databricks/appkit";
import { agent, chat } from "@databricks/appkit-agent";

await createApp({
  plugins: [
    server({ autoStart: true }),
    agent({
      model: "databricks-claude-sonnet-4-5",
      systemPrompt: "You are a helpful assistant.",
    }),
    chat({
      backend: "agent",
    }),
  ],
});
```

The agent plugin registers `POST /api/agent` (OpenAI Responses API format with SSE streaming). The chat plugin registers routes under `/api/chat/` for streaming chat, history, feedback, and more.

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

  // Or bring your own AgentInterface implementation
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
  type: "genie_space" as const,
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

The `StandardAgent` class (exported from this package) is the built-in implementation used when you pass `model` instead of `agentInstance`. It translates the underlying agent's stream events into Responses API format.

## Chat Plugin

The chat plugin provides a streaming chat server that can be wired to the agent plugin or to a remote Databricks serving endpoint.

### Configuration

```typescript
import { chat } from "@databricks/appkit-agent";

chat({
  // Route chat through the local agent plugin
  backend: "agent",

  // Or point to a remote proxy / serving endpoint:
  // backend: { proxy: "http://localhost:8000/invocations" },
  // backend: { endpoint: "databricks-claude-sonnet-4-5" },

  // Optional: PostgreSQL pool for persistent chat history
  // pool: new Pool(),

  // Auto-create the ai_chatbot schema and tables on startup (default: false)
  // autoMigrate: true,

  // Enable thumbs up/down feedback (default: !!process.env.MLFLOW_EXPERIMENT_ID)
  // feedbackEnabled: true,

  // Custom session resolver (default: x-forwarded-user headers → SCIM → env)
  // getSession: (req) => ({ user: { id: req.headers["x-user-id"] } }),
});
```

### Persistence

Without a `pool`, the chat plugin runs in **ephemeral mode** — conversations are not saved. To enable persistent chat history, pass a `pg.Pool`:

```typescript
import { Pool } from "pg";

chat({
  backend: "agent",
  pool: new Pool({ connectionString: "postgres://..." }),
  autoMigrate: true,
});
```

When `autoMigrate: true` is set, the plugin creates the `ai_chatbot` schema and tables (`Chat`, `Message`, `Vote`) on startup using [Drizzle migrations](https://orm.drizzle.team/docs/migrations). Migration SQL is generated from the Drizzle schema definition, so it stays in sync automatically.

### Session Resolution

In production on Databricks Apps, sessions are resolved from headers set by the platform proxy (`x-forwarded-user`, `x-forwarded-email`). In local development, the plugin falls back to the SCIM `/Me` API (if Databricks auth is configured) or `process.env.USER`.

You can override this entirely with a custom `getSession` callback.

### Feedback

When feedback is enabled, thumbs up/down votes are submitted as assessments to the [MLflow Traces API](https://docs.databricks.com/en/mlflow/llm-tracing.html). Votes are also persisted in the database when a pool is configured.

### Chat API Routes

All routes are mounted under `/api/chat/`.

| Method   | Path                     | Auth         | Description                        |
| -------- | ------------------------ | ------------ | ---------------------------------- |
| `GET`    | `/config`                | —            | Feature flags (history, feedback)  |
| `GET`    | `/session`               | —            | Current user session               |
| `GET`    | `/history`               | required     | Paginated chat list                |
| `GET`    | `/messages/:id`          | required+ACL | Messages for a chat                |
| `DELETE` | `/messages/:id/trailing` | required     | Delete messages after a given one  |
| `POST`   | `/feedback`              | required     | Submit thumbs up/down              |
| `GET`    | `/feedback/chat/:chatId` | required+ACL | Votes for a chat                   |
| `POST`   | `/title`                 | required     | Auto-generate title from message   |
| `PATCH`  | `/:id/visibility`        | required+ACL | Toggle public/private              |
| `GET`    | `/:id/stream`            | required     | Resume an active stream            |
| `GET`    | `/:id`                   | required+ACL | Get single chat                    |
| `DELETE` | `/:id`                   | required+ACL | Delete a chat                      |
| `POST`   | `/`                      | required     | Main chat handler (streaming)      |

## API Reference

### Exports

| Export                | Kind           | Description                                                |
| --------------------- | -------------- | ---------------------------------------------------------- |
| `agent`               | Plugin factory | Agent plugin — call with config, pass to `createApp`       |
| `chat`                | Plugin factory | Chat plugin — call with config, pass to `createApp`        |
| `ChatPlugin`          | Class          | Chat plugin class (for `ChatPlugin.staticAssetsPath`, etc) |
| `StandardAgent`       | Class          | Built-in `AgentInterface` implementation                   |
| `createInvokeHandler` | Function       | Express handler factory for the `/api/agent` endpoint      |
| `isFunctionTool`      | Function       | Type guard for `FunctionTool`                              |
| `isHostedTool`        | Function       | Type guard for `HostedTool`                                |

### Types

| Type                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `IAgentConfig`        | Agent plugin configuration options                            |
| `ChatConfig`          | Chat plugin configuration options                             |
| `ChatBackend`         | Backend target: plugin name, `{ proxy }`, or `{ endpoint }`  |
| `ChatSession`         | Session object with user info                                 |
| `GetSession`          | Custom session resolver function type                         |
| `AgentInterface`      | Contract for custom agent implementations                     |
| `AgentTool`           | Union of `FunctionTool \| HostedTool`                         |
| `FunctionTool`        | Local tool with JSON Schema parameters and `execute` handler  |
| `HostedTool`          | Union of Genie, Vector Search, Custom MCP, External MCP tools |
| `InvokeParams`        | Input to `invoke()` / `stream()`                              |
| `ResponseOutputItem`  | Output item (message, function call, or function call output) |
| `ResponseStreamEvent` | SSE event types for streaming responses                       |

## License

See [LICENSE](./LICENSE).
