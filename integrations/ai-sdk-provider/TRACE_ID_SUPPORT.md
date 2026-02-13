# Trace ID Support for Feedback Collection

## Overview

This package now supports requesting trace IDs from Databricks agent serving endpoints via the `databricks_options.return_trace` parameter. This enables feedback submission to MLflow using the trace IDs.

## Changes Made

### 1. Request Body Support
- Added `databricks_options` field to `ResponsesBodyArgs` type
- Added `databricksOptions` field to `DatabricksProviderOptions` type
- Implemented logic to merge `databricks_options` into API requests

### 2. Response Schema Support
- Updated `responsesAgentResponseSchema` to include optional `trace_id` and `span_id` fields
- Updated `responsesCompletedSchema` (streaming) to include optional `trace_id` and `span_id` fields

### 3. Tests
- Created comprehensive tests in `tests/databricks-options.test.ts`
- All 5 new tests pass
- All 287 existing tests continue to pass

## Usage

### Basic Example

```typescript
import { createDatabricksProvider } from '@databricks/ai-sdk-provider'
import { streamText } from 'ai'

const provider = createDatabricksProvider({
  baseURL: 'https://your-workspace.databricks.com/serving-endpoints',
  headers: {
    Authorization: `Bearer ${token}`,
  },
})

const model = provider.responses('your-agent-endpoint')

let traceId: string | undefined
let spanId: string | undefined

const result = await streamText({
  model,
  messages: [{ role: 'user', content: 'Hello!' }],
  providerOptions: {
    databricks: {
      databricksOptions: {
        return_trace: true,  // Request trace ID
      },
    },
  },
  onFinish: ({ rawResponse }) => {
    // Extract trace ID from response
    traceId = rawResponse?.trace_id
    spanId = rawResponse?.span_id

    console.log('Trace ID:', traceId)
    console.log('Span ID:', spanId)

    // Use for feedback submission to MLflow
  },
})
```

### TypeScript Types

The types are fully typed and will provide autocomplete:

```typescript
import type { DatabricksProviderOptions } from '@databricks/ai-sdk-provider'

const options: DatabricksProviderOptions = {
  databricksOptions: {
    return_trace: true,  // ✅ Type-safe
  },
}
```

### Combining with Other Options

You can use `databricks_options` alongside other provider options:

```typescript
providerOptions: {
  databricks: {
    parallelToolCalls: true,
    metadata: {
      user_id: '123',
      session_id: 'abc',
    },
    reasoning: {
      effort: 'high',
    },
    databricksOptions: {
      return_trace: true,
    },
  },
}
```

### Feedback Submission Example

Once you have the trace ID, you can submit feedback to MLflow:

```typescript
import { streamText } from 'ai'

let traceId: string | undefined

const result = await streamText({
  model,
  messages: [...],
  providerOptions: {
    databricks: {
      databricksOptions: {
        return_trace: true,
      },
    },
  },
  onFinish: async ({ rawResponse, responseMessage }) => {
    traceId = rawResponse?.trace_id

    // Save message with trace ID
    await saveMessage({
      id: responseMessage.id,
      content: responseMessage.content,
      traceId,  // Store for later feedback
    })
  },
})

// Later, when user provides feedback
async function submitFeedback(messageId: string, feedback: 'thumbs_up' | 'thumbs_down') {
  const message = await getMessage(messageId)

  if (message.traceId) {
    await fetch(`${databricksHost}/api/2.0/mlflow/traces/assessments`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: 'user_feedback',
        trace_id: message.traceId,  // Use real trace ID
        value: feedback === 'thumbs_up' ? 1 : 0,
        value_type: 'numeric',
        tags: {
          feedback_type: feedback,
        },
      }),
    })
  }
}
```

## API Reference

### DatabricksProviderOptions

```typescript
type DatabricksProviderOptions = {
  parallelToolCalls?: boolean
  metadata?: Record<string, string>
  reasoning?: {
    effort?: 'low' | 'medium' | 'high'
  }
  databricksOptions?: {
    return_trace?: boolean  // NEW: Request trace ID from endpoint
  }
}
```

### Response Schema

Non-streaming responses (`doGenerate`):
```typescript
{
  id: string
  model?: string
  output: Array<...>
  usage?: { input_tokens, output_tokens, total_tokens }
  trace_id?: string    // NEW: Trace ID if requested
  span_id?: string     // NEW: Span ID if requested
}
```

Streaming responses (`doStream` - in final `responses.completed` event):
```typescript
{
  type: 'responses.completed'
  response: {
    id: string
    status?: 'completed' | 'failed' | ...
    usage: { input_tokens, output_tokens, total_tokens }
    trace_id?: string    // NEW: Trace ID if requested
    span_id?: string     // NEW: Span ID if requested
  }
}
```

## Compatibility

- ✅ **Backward Compatible**: All changes are optional
- ✅ **Type Safe**: Full TypeScript support with autocomplete
- ✅ **Tested**: 5 new tests + 287 existing tests passing
- ✅ **No Breaking Changes**: Existing code continues to work

## Related Documentation

- [Databricks MLflow Tracing](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/)
- [Collect User Feedback](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/collect-user-feedback)
- [Query Agent Documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/query-agent)
- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs)

## Testing

Run the test suite:
```bash
npm test
```

Run only the databricks_options tests:
```bash
npm test databricks-options
```

## Building

```bash
npm run build
```

## Next Steps

To use this functionality:

1. Build and publish the package (or use local package)
2. Update your application to pass `providerOptions.databricks.databricksOptions.return_trace = true`
3. Extract `trace_id` from `rawResponse` in callbacks
4. Store trace IDs with messages
5. Use trace IDs for feedback submission to MLflow
