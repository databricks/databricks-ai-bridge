# Fix: Make `annotations` field optional in responses-agent-schema

## Problem

The `annotations` field in `responsesAgentMessageSchema` is defined as a **required** array, but most MLflow agent responses don't include this field, causing a type validation error:

```
AI_TypeValidationError: Type validation failed
path: ["output",0,"content",0,"annotations"]
expected: array
received: undefined
```

## Root Cause

In `integrations/ai-sdk-provider/src/responses-agent-language-model/responses-agent-schema.ts`, line 15:

```typescript
annotations: z.array(
  z.discriminatedUnion('type', [
    z.object({
      type: z.literal('url_citation'),
      start_index: z.number(),
      end_index: z.number(),
      url: z.string(),
      title: z.string(),
    }),
  ])
),
```

This is used by:
1. `responsesAgentResponseSchema` (non-streaming responses)
2. `responseOutputItemDoneSchema` (streaming `response.output_item.done` events)

## Solution

Make the `annotations` field optional with a default empty array:

```typescript
annotations: z.array(
  z.discriminatedUnion('type', [
    z.object({
      type: z.literal('url_citation'),
      start_index: z.number(),
      end_index: z.number(),
      url: z.string(),
      title: z.string(),
    }),
  ])
).optional().default([]),
```

## Files to Modify

1. `integrations/ai-sdk-provider/src/responses-agent-language-model/responses-agent-schema.ts`
   - Line 15-25: Add `.optional().default([])` to annotations field

## Testing

1. Run existing tests to ensure no regressions:
   ```bash
   cd integrations/ai-sdk-provider && npm test
   ```

2. Test with a backend that doesn't return `annotations` field

## Version Bump

After the fix, bump the version in `integrations/ai-sdk-provider/package.json` (currently 0.4.1 → 0.4.2)
