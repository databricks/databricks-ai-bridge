# Implementation Plan: Add `databricks_options.return_trace` Support

## Overview
Add support for passing `databricks_options.return_trace = true` to Databricks agent serving endpoints to enable trace ID retrieval for feedback submission.

## Current State

The AI SDK provider already has infrastructure for provider-specific options:
- ✅ `providerOptions.databricks` mechanism exists
- ✅ Options are extracted in `callOptionsToResponsesArgs()`
- ✅ Options are merged into request body

**Missing:**
- ❌ `databricks_options` not defined in `DatabricksProviderOptions`
- ❌ `databricks_options` not added to request body
- ❌ `trace_id` and `span_id` not in response schema

## Files to Modify

### 1. `integrations/ai-sdk-provider/src/responses-agent-language-model/call-options-to-responses-args.ts`

**Add to `ResponsesBodyArgs` type (line 7-22):**
```typescript
export type ResponsesBodyArgs = {
  max_output_tokens?: number
  temperature?: number
  top_p?: number
  parallel_tool_calls?: boolean
  metadata?: Record<string, string>
  reasoning?: {
    effort?: 'low' | 'medium' | 'high'
  }
  text?: {
    format?:
      | { type: 'text' }
      | { type: 'json_object' }
      | { type: 'json_schema'; json_schema: unknown }
  }
  // Add this:
  databricks_options?: {
    return_trace?: boolean
  }
}
```

**Add to `DatabricksProviderOptions` type (line 27-33):**
```typescript
export type DatabricksProviderOptions = {
  parallelToolCalls?: boolean
  metadata?: Record<string, string>
  reasoning?: {
    effort?: 'low' | 'medium' | 'high'
  }
  // Add this:
  databricksOptions?: {
    return_trace?: boolean
  }
}
```

**Add to `callOptionsToResponsesArgs` function (after line 148):**
```typescript
  if (databricksOptions?.databricksOptions != null) {
    args.databricks_options = databricksOptions.databricksOptions
  }

  return { args, warnings }
}
```

### 2. `integrations/ai-sdk-provider/src/responses-agent-language-model/responses-agent-schema.ts`

**Update `responsesAgentResponseSchema` (line 81-104):**
```typescript
export const responsesAgentResponseSchema = z.object({
  id: z.string().optional(),
  created_at: z.number().optional(),
  error: z
    .object({
      code: z.string(),
      message: z.string(),
    })
    .nullish(),
  model: z.string().optional(),
  output: z.array(responsesAgentOutputItem),
  incomplete_details: z
    .object({
      reason: z.string().nullish().optional(),
    })
    .nullish(),
  usage: z
    .object({
      input_tokens: z.number(),
      output_tokens: z.number(),
      total_tokens: z.number(),
    })
    .optional(),
  // Add these fields:
  trace_id: z.string().optional(),
  span_id: z.string().optional(),
})
```

**Update `responsesCompletedSchema` (line 168-186):**
```typescript
const responsesCompletedSchema = z.object({
  type: z.literal('responses.completed'),
  response: z.object({
    id: z.string(),
    status: z
      .enum(['completed', 'failed', 'in_progress', 'cancelled', 'queued', 'incomplete'])
      .optional(),
    incomplete_details: z
      .object({
        reason: z.string().nullish().optional(),
      })
      .nullish(),
    usage: z.object({
      input_tokens: z.number(),
      output_tokens: z.number(),
      total_tokens: z.number(),
    }),
    // Add these fields:
    trace_id: z.string().optional(),
    span_id: z.string().optional(),
  }),
})
```

### 3. Update TypeScript exports

The response types are already exported, so `trace_id` and `span_id` will automatically be available in the TypeScript types.

## Testing Changes

Create a test file: `integrations/ai-sdk-provider/tests/databricks-options.test.ts`

```typescript
import { describe, it, expect } from 'vitest'
import { callOptionsToResponsesArgs } from '../src/responses-agent-language-model/call-options-to-responses-args'

describe('databricks_options support', () => {
  it('should include databricks_options when provided', () => {
    const result = callOptionsToResponsesArgs({
      prompt: [],
      providerOptions: {
        databricks: {
          databricksOptions: {
            return_trace: true,
          },
        },
      },
    })

    expect(result.args.databricks_options).toEqual({
      return_trace: true,
    })
  })

  it('should not include databricks_options when not provided', () => {
    const result = callOptionsToResponsesArgs({
      prompt: [],
      providerOptions: {},
    })

    expect(result.args.databricks_options).toBeUndefined()
  })
})
```

## Building and Publishing

```bash
cd ~/databricks-ai-bridge/integrations/ai-sdk-provider

# Install dependencies
npm install

# Run tests
npm test

# Build
npm run build

# Bump version (patch for bug fix, minor for feature)
npm version minor

# Publish to npm (if you have permissions)
npm publish

# Or create a local package for testing
npm pack
```

## Using in e2e-chatbot-app-next

After publishing the new version:

```bash
cd ~/app-templates/e2e-chatbot-app-next
npm install @databricks/ai-sdk-provider@latest
```

Then update `server/src/routes/chat.ts`:

```typescript
let responseTraceId: string | undefined;
let responseSpanId: string | undefined;

const result = streamText({
  model,
  messages: convertToModelMessages(uiMessages),
  providerOptions: {
    databricks: {
      databricksOptions: {
        return_trace: true,  // Request trace ID
      },
    },
  },
  onFinish: ({ usage, rawResponse }) => {
    finalUsage = usage;

    // Extract trace ID from response
    responseTraceId = (rawResponse as any)?.trace_id;
    responseSpanId = (rawResponse as any)?.span_id;

    console.log('[Trace] Got trace ID:', responseTraceId);
    console.log('[Trace] Got span ID:', responseSpanId);
  },
  tools: {
    [DATABRICKS_TOOL_CALL_ID]: DATABRICKS_TOOL_DEFINITION,
  },
});
```

## Documentation to Update

Update `integrations/ai-sdk-provider/README.md`:

```markdown
### Requesting Trace IDs for Feedback

To retrieve trace IDs for feedback submission with MLflow, pass `databricksOptions.return_trace`:

\`\`\`typescript
const result = await streamText({
  model: provider.responses('my-agent'),
  messages: [...],
  providerOptions: {
    databricks: {
      databricksOptions: {
        return_trace: true,
      },
    },
  },
  onFinish: ({ rawResponse }) => {
    const traceId = rawResponse.trace_id
    const spanId = rawResponse.span_id
    // Use these for feedback submission
  },
})
\`\`\`

The trace_id and span_id will be included in the response and can be used to log feedback to MLflow.
```

## Changelog Entry

Add to `integrations/ai-sdk-provider/CHANGELOG.md`:

```markdown
## [0.3.0] - YYYY-MM-DD

### Added
- Support for `providerOptions.databricks.databricksOptions.return_trace` to request trace IDs from agent serving endpoints
- Response schemas now include optional `trace_id` and `span_id` fields
- Trace IDs can be used for feedback submission to MLflow

### Example
\`\`\`typescript
providerOptions: {
  databricks: {
    databricksOptions: {
      return_trace: true,
    },
  },
}
\`\`\`
```

## Summary

**Changes Required:**
1. ✏️ Add `databricks_options` to request body types
2. ✏️ Add `databricksOptions` to provider options type
3. ✏️ Add logic to merge `databricks_options` into request
4. ✏️ Add `trace_id` and `span_id` to response schemas
5. ✅ Write tests
6. ✅ Update documentation
7. ✅ Build and publish

**Complexity:** Low - following existing patterns
**Breaking Changes:** None - all additions are optional
**Testing:** Unit tests + manual E2E test with chatbot app
