import type { LanguageModelV2StreamPart } from '@ai-sdk/provider'

/**
 * FMAPI output fixtures for testing.
 * These represent SSE streams from the FMAPI endpoint.
 * FMAPI uses XML tags for tool calls: <tool_call> and <tool_call_result>
 */

type LLMOutputFixtures = {
  in: string
  out: Array<LanguageModelV2StreamPart>
}

/**
 * Basic text streaming output from FMAPI
 */
export const FMAPI_BASIC_TEXT_OUTPUT: LLMOutputFixtures = {
  in: `
data: {
  "id": "fmapi-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hello! "
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "I'm an AI assistant. "
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "How can I help you today?"
      },
      "finish_reason": null
    }
  ]
}
`,
  out: [
    { type: 'text-start', id: 'fmapi-123' },
    {
      type: 'text-delta',
      id: 'fmapi-123',
      delta: 'Hello! ',
    },
    {
      type: 'text-delta',
      id: 'fmapi-123',
      delta: "I'm an AI assistant. ",
    },
    {
      type: 'text-delta',
      id: 'fmapi-123',
      delta: 'How can I help you today?',
    },
    { type: 'text-end', id: 'fmapi-123' },
  ],
}

/**
 * FMAPI output with tool calls using XML tags
 */
export const FMAPI_WITH_TOOL_CALLS: LLMOutputFixtures = {
  in: `
data: {
  "id": "fmapi-456",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Let me check the weather for you. "
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-456",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "<tool_call>{\\"id\\": \\"call_weather_001\\", \\"name\\": \\"get_weather\\", \\"arguments\\": {\\"location\\": \\"New York\\"}}</tool_call>"
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-456",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "<tool_call_result>{\\"id\\": \\"call_weather_001\\", \\"content\\": {\\"temperature\\": 22, \\"condition\\": \\"sunny\\"}}</tool_call_result>"
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-456-response",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "The weather in New York is 22°C and sunny."
      },
      "finish_reason": null
    }
  ]
}
`,
  out: [
    { type: 'text-start', id: 'fmapi-456' },
    {
      type: 'text-delta',
      id: 'fmapi-456',
      delta: 'Let me check the weather for you. ',
    },
    { type: 'text-end', id: 'fmapi-456' },
    {
      type: 'tool-call',
      toolCallId: 'call_weather_001',
      toolName: 'get_weather',
      input: '{"location":"New York"}',
      providerExecuted: true,
    },
    {
      type: 'tool-result',
      toolCallId: 'call_weather_001',
      toolName: 'databricks-tool-call',
      result: { temperature: 22, condition: 'sunny' },
    },
    { type: 'text-start', id: 'fmapi-456-response' },
    {
      type: 'text-delta',
      id: 'fmapi-456-response',
      delta: 'The weather in New York is 22°C and sunny.',
    },
    { type: 'text-end', id: 'fmapi-456-response' },
  ],
}

/**
 * FMAPI output with OpenAI-format streaming tool_calls
 * OpenAI sends tool call ID only in the first chunk, subsequent chunks use index
 */
export const FMAPI_WITH_OPENAI_STREAMING_TOOL_CALLS: LLMOutputFixtures = {
  in: `
data: {
  "id": "fmapi-openai-tools",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "index": 0,
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": ""
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-openai-tools",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "function": {
              "arguments": "{\\"location\\":"
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-openai-tools",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "function": {
              "arguments": "\\"San Francisco\\"}"
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-openai-tools",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {},
      "finish_reason": "tool_calls"
    }
  ]
}
`,
  out: [
    // tool-input-start emitted when tool call begins
    { type: 'tool-input-start', id: 'call_abc123', toolName: 'get_weather' },
    // tool-input-delta emitted for each argument chunk
    { type: 'tool-input-delta', id: 'call_abc123', delta: '{"location":' },
    { type: 'tool-input-delta', id: 'call_abc123', delta: '"San Francisco"}' },
    // tool-input-end emitted in flush
    { type: 'tool-input-end', id: 'call_abc123' },
    // Complete tool-call emitted in flush
    {
      type: 'tool-call',
      toolCallId: 'call_abc123',
      toolName: 'get_weather',
      input: '{"location":"San Francisco"}',
    },
  ],
}

/**
 * FMAPI output with multiple parallel OpenAI-format streaming tool_calls
 */
export const FMAPI_WITH_PARALLEL_STREAMING_TOOL_CALLS: LLMOutputFixtures = {
  in: `
data: {
  "id": "fmapi-parallel",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "index": 0,
            "id": "call_tool_a",
            "type": "function",
            "function": {
              "name": "tool_a",
              "arguments": ""
            }
          },
          {
            "index": 1,
            "id": "call_tool_b",
            "type": "function",
            "function": {
              "name": "tool_b",
              "arguments": ""
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-parallel",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "function": {
              "arguments": "{\\"x\\":1}"
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-parallel",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 1,
            "function": {
              "arguments": "{\\"y\\":2}"
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-parallel",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {},
      "finish_reason": "tool_calls"
    }
  ]
}
`,
  out: [
    // First chunk starts both tools
    { type: 'tool-input-start', id: 'call_tool_a', toolName: 'tool_a' },
    { type: 'tool-input-start', id: 'call_tool_b', toolName: 'tool_b' },
    // Arguments stream in (tool_a)
    { type: 'tool-input-delta', id: 'call_tool_a', delta: '{"x":1}' },
    // Arguments stream in (tool_b) - uses tracked ID from index
    { type: 'tool-input-delta', id: 'call_tool_b', delta: '{"y":2}' },
    // Flush emits end and complete tool-call events
    { type: 'tool-input-end', id: 'call_tool_a' },
    {
      type: 'tool-call',
      toolCallId: 'call_tool_a',
      toolName: 'tool_a',
      input: '{"x":1}',
    },
    { type: 'tool-input-end', id: 'call_tool_b' },
    {
      type: 'tool-call',
      toolCallId: 'call_tool_b',
      toolName: 'tool_b',
      input: '{"y":2}',
    },
  ],
}

/**
 * FMAPI output with legacy UC function call tags
 */
export const FMAPI_WITH_LEGACY_TAGS: LLMOutputFixtures = {
  in: `
data: {
  "id": "fmapi-789",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "I'll execute that calculation. "
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-789",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "<uc_function_call>{\\"id\\": \\"calc_001\\", \\"name\\": \\"calculate\\", \\"arguments\\": {\\"expression\\": \\"2 + 2\\"}}</uc_function_call>"
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-789",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "<uc_function_result>{\\"id\\": \\"calc_001\\", \\"content\\": \\"4\\"}</uc_function_result>"
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "fmapi-789-response",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "databricks-meta-llama-3-1-405b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "The result is 4."
      },
      "finish_reason": null
    }
  ]
}
`,
  out: [
    { type: 'text-start', id: 'fmapi-789' },
    {
      type: 'text-delta',
      id: 'fmapi-789',
      delta: "I'll execute that calculation. ",
    },
    { type: 'text-end', id: 'fmapi-789' },
    {
      type: 'tool-call',
      toolCallId: 'calc_001',
      toolName: 'calculate',
      input: '{"expression":"2 + 2"}',
      providerExecuted: true,
    },
    {
      type: 'tool-result',
      toolCallId: 'calc_001',
      toolName: 'databricks-tool-call',
      result: '4',
    },
    { type: 'text-start', id: 'fmapi-789-response' },
    {
      type: 'text-delta',
      id: 'fmapi-789-response',
      delta: 'The result is 4.',
    },
    { type: 'text-end', id: 'fmapi-789-response' },
  ],
}
