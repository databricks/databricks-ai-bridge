import type { LanguageModelV3StreamPart } from '@ai-sdk/provider'

/**
 * FMAPI output fixtures for testing.
 * These represent SSE streams and JSON responses from the FMAPI endpoint.
 */

type LLMOutputFixtures = {
  in: string
  out: Array<LanguageModelV3StreamPart>
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
 * Non-streaming FMAPI response with tool_calls (for doGenerate tests)
 */
export const FMAPI_RESPONSE_WITH_TOOL_CALLS = {
  id: 'fmapi-response-tools',
  object: 'chat.completion',
  created: 1234567890,
  model: 'databricks-meta-llama-3-1-405b-instruct',
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_weather_123',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"location":"New York","unit":"celsius"}',
            },
          },
        ],
      },
      finish_reason: 'tool_calls',
    },
  ],
  usage: {
    prompt_tokens: 50,
    completion_tokens: 25,
    total_tokens: 75,
  },
}

/**
 * Non-streaming FMAPI response with multiple parallel tool_calls
 */
export const FMAPI_RESPONSE_WITH_PARALLEL_TOOL_CALLS = {
  id: 'fmapi-response-parallel-tools',
  object: 'chat.completion',
  created: 1234567890,
  model: 'databricks-meta-llama-3-1-405b-instruct',
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_tool_1',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"location":"Paris"}',
            },
          },
          {
            id: 'call_tool_2',
            type: 'function',
            function: {
              name: 'get_time',
              arguments: '{"timezone":"Europe/Paris"}',
            },
          },
        ],
      },
      finish_reason: 'tool_calls',
    },
  ],
  usage: {
    prompt_tokens: 60,
    completion_tokens: 40,
    total_tokens: 100,
  },
}
