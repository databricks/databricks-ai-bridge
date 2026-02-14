# Implementation Summary: Trace ID Support

## Overview

Successfully implemented support for passing `databricks_options.return_trace = true` to Databricks Responses API endpoints and exposing `trace_id` and `span_id` in the response for feedback collection.

## Changes Made

### 1. Request Body Support (Already Done in WIP)
- ✅ Added `databricks_options` field to `ResponsesBodyArgs` type
- ✅ Added `databricksOptions` field to `DatabricksProviderOptions` type
- ✅ Implemented logic to merge `databricks_options` into API requests
- Location: `src/responses-agent-language-model/call-options-to-responses-args.ts`

### 2. Response Schema Support (Already Done in WIP)
- ✅ Updated `responsesAgentResponseSchema` to include optional `trace_id` and `span_id` fields
- ✅ Updated `responsesCompletedSchema` (streaming) to include optional `trace_id` and `span_id` fields
- Location: `src/responses-agent-language-model/responses-agent-schema.ts`

### 3. Response Body Exposure (New Implementation)
- ✅ **Non-streaming (`doGenerate`)**: Added `response.body` containing the full API response including `trace_id` and `span_id`
- ✅ **Streaming (`doStream`)**: Captured `trace_id` and `span_id` from `responses.completed` event and exposed them in `response.body`
- Location: `src/responses-agent-language-model/responses-agent-language-model.ts`

### 4. Tests
- ✅ 5 tests for request body options (already passing)
- ✅ 3 new tests for response body exposure (newly added and passing)
- Locations:
  - `tests/databricks-options.test.ts` (request tests)
  - `tests/trace-id-response.test.ts` (response tests - NEW)

### 5. Documentation
- ✅ Updated `CHANGELOG.md` with the new feature
- ✅ Created example usage in `examples/trace-id-example.ts`
- ✅ Existing documentation in `TRACE_ID_SUPPORT.md` (from WIP)

## Key Implementation Details

### Streaming Challenge
The main challenge was that the `trace_id` and `span_id` are only available after the stream completes (in the `responses.completed` event), but the response object is returned immediately when `doStream()` is called.

**Solution**: Created a mutable `responseBody` object that gets populated as the stream is consumed. Since JavaScript objects are passed by reference, the same object reference is returned immediately but gets populated with `trace_id` and `span_id` when the `responses.completed` chunk arrives.

```typescript
// Create mutable object
const responseBody: Record<string, unknown> = {}

// Populate it when responses.completed arrives
if (chunk.value.type === 'responses.completed') {
  if (chunk.value.response.trace_id !== undefined) {
    responseBody.trace_id = chunk.value.response.trace_id
  }
  if (chunk.value.response.span_id !== undefined) {
    responseBody.span_id = chunk.value.response.span_id
  }
}

// Return the object reference immediately
return {
  stream: ...,
  response: {
    body: responseBody,  // Will be populated as stream is consumed
  },
}
```

## Usage Example

```typescript
import { createDatabricksProvider } from '@databricks/ai-sdk-provider'
import { streamText } from 'ai'

const provider = createDatabricksProvider({ ... })
const model = provider.responses('your-agent-endpoint')

let traceId: string | undefined

const result = await streamText({
  model,
  messages: [...],
  providerOptions: {
    databricks: {
      databricksOptions: {
        return_trace: true,  // Request trace ID
      },
    },
  },
  onFinish: ({ response }) => {
    // Access trace_id from response.body
    const rawResponse = response?.body as any
    traceId = rawResponse?.trace_id

    // Use for MLflow feedback submission
    console.log('Trace ID:', traceId)
  },
})
```

## Test Results

All tests passing:
- ✅ 292 existing tests
- ✅ 5 databricks_options tests (request body)
- ✅ 3 trace-id-response tests (response body)
- **Total: 300 tests passing**

## Build Status

✅ Package builds successfully with no errors

## Backward Compatibility

✅ **No breaking changes** - all additions are optional:
- `databricksOptions.return_trace` is optional
- `trace_id` and `span_id` are optional in response
- Existing code continues to work without changes

## Next Steps

1. ✅ Implementation complete
2. ✅ Tests passing
3. ✅ Documentation updated
4. ⏭️ Ready for version bump and publish
5. ⏭️ Update consuming applications to use the feature

## Files Modified

1. `src/responses-agent-language-model/call-options-to-responses-args.ts` (already done in WIP)
2. `src/responses-agent-language-model/responses-agent-schema.ts` (already done in WIP)
3. `src/responses-agent-language-model/responses-agent-language-model.ts` (NEW - response body exposure)
4. `tests/databricks-options.test.ts` (already done in WIP)
5. `tests/trace-id-response.test.ts` (NEW)
6. `CHANGELOG.md` (NEW)
7. `examples/trace-id-example.ts` (NEW)

## Related Documentation

- [TRACE_ID_SUPPORT.md](./TRACE_ID_SUPPORT.md) - User-facing documentation
- [TRACE_ID_IMPLEMENTATION_PLAN.md](../../TRACE_ID_IMPLEMENTATION_PLAN.md) - Original plan
- [Databricks MLflow Tracing](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/)
- [Vercel AI SDK](https://sdk.vercel.ai/docs)
