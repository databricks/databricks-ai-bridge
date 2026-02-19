import { describe, expect, it } from 'vitest';
import {
  openResponsesChunkSchema,
  openResponsesTextDeltaSchema,
  openResponsesItemDoneSchema,
  openResponsesErrorSchema,
} from '../src/open-responses-language-model/open-responses-schema';
import { convertOpenResponsesChunkToStreamPart } from '../src/open-responses-language-model/open-responses-convert-to-message-parts';

describe('OpenResponses Schema Validation', () => {
  it('validates text delta event', () => {
    const event = {
      type: 'response.output_text.delta',
      delta: 'Hello',
    };
    const result = openResponsesTextDeltaSchema.safeParse(event);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.delta).toBe('Hello');
    }
  });

  it('validates item done event', () => {
    const event = {
      type: 'response.output_item.done',
      item: {
        role: 'assistant',
        content: 'Hello world',
      },
    };
    const result = openResponsesItemDoneSchema.safeParse(event);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.item.content).toBe('Hello world');
    }
  });

  it('validates error event', () => {
    const event = {
      type: 'response.error',
      error: {
        message: 'Something went wrong',
        type: 'internal_error',
      },
    };
    const result = openResponsesErrorSchema.safeParse(event);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.error.message).toBe('Something went wrong');
    }
  });

  it('validates error event without type field', () => {
    const event = {
      type: 'response.error',
      error: {
        message: 'Error without type',
      },
    };
    const result = openResponsesErrorSchema.safeParse(event);
    expect(result.success).toBe(true);
  });

  it('validates with discriminated union', () => {
    const event1 = {
      type: 'response.output_text.delta',
      delta: 'test',
    };
    const result1 = openResponsesChunkSchema.safeParse(event1);
    expect(result1.success).toBe(true);

    const event2 = {
      type: 'response.output_item.done',
      item: { role: 'assistant', content: 'done' },
    };
    const result2 = openResponsesChunkSchema.safeParse(event2);
    expect(result2.success).toBe(true);

    const event3 = {
      type: 'response.error',
      error: { message: 'error' },
    };
    const result3 = openResponsesChunkSchema.safeParse(event3);
    expect(result3.success).toBe(true);
  });

  it('rejects invalid event types', () => {
    const event = {
      type: 'invalid.type',
      data: 'something',
    };
    const result = openResponsesChunkSchema.safeParse(event);
    expect(result.success).toBe(false);
  });
});

describe('OpenResponses Conversion to Stream Parts', () => {
  it('converts text delta to stream part', () => {
    const chunk = {
      type: 'response.output_text.delta' as const,
      delta: 'Hello',
    };
    const result = convertOpenResponsesChunkToStreamPart(chunk);
    expect(result).toEqual({
      type: 'text-delta',
      textDelta: 'Hello',
    });
  });

  it('converts item done to finish event', () => {
    const chunk = {
      type: 'response.output_item.done' as const,
      item: {
        role: 'assistant' as const,
        content: 'Hello world',
      },
    };
    const result = convertOpenResponsesChunkToStreamPart(chunk);
    expect(result).toEqual({
      type: 'finish',
      finishReason: 'stop',
      usage: {
        promptTokens: 0,
        completionTokens: 0,
      },
      providerMetadata: {
        databricks: {
          format: 'openresponses',
          fullContent: 'Hello world',
        },
      },
    });
  });

  it('converts error to error stream part', () => {
    const chunk = {
      type: 'response.error' as const,
      error: {
        message: 'Something went wrong',
        type: 'internal_error',
      },
    };
    const result = convertOpenResponsesChunkToStreamPart(chunk);
    expect(result).toMatchObject({
      type: 'error',
      error: expect.any(Error),
    });
    if (result && result.type === 'error') {
      expect(result.error.message).toBe('Something went wrong');
    }
  });

  it('handles multiple text deltas', () => {
    const chunks = [
      { type: 'response.output_text.delta' as const, delta: 'Hello' },
      { type: 'response.output_text.delta' as const, delta: ' ' },
      { type: 'response.output_text.delta' as const, delta: 'world' },
    ];

    const results = chunks.map(convertOpenResponsesChunkToStreamPart);
    expect(results).toHaveLength(3);
    expect(results[0]).toEqual({ type: 'text-delta', textDelta: 'Hello' });
    expect(results[1]).toEqual({ type: 'text-delta', textDelta: ' ' });
    expect(results[2]).toEqual({ type: 'text-delta', textDelta: 'world' });
  });
});
