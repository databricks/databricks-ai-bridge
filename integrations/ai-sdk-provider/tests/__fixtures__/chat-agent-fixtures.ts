import type { LanguageModelV3StreamPart } from '@ai-sdk/provider'

/**
 * Chat Agent output fixtures for testing.
 * These represent SSE streams from the Chat Agent endpoint.
 */

type LLMOutputFixtures = {
  in: string
  out: Array<LanguageModelV3StreamPart>
}

/**
 * Basic text streaming output from Chat Agent
 */
export const CHAT_AGENT_BASIC_TEXT_OUTPUT: LLMOutputFixtures = {
  in: `
data: {
  "id": "chatcmpl-123",
  "delta": {
    "role": "assistant",
    "content": "Hello",
    "id": "msg_001"
  }
}

data: {
  "id": "chatcmpl-123",
  "delta": {
    "role": "assistant",
    "content": " there",
    "id": "msg_001"
  }
}

data: {
  "id": "chatcmpl-123",
  "delta": {
    "role": "assistant",
    "content": "! How can",
    "id": "msg_001"
  }
}

data: {
  "id": "chatcmpl-123",
  "delta": {
    "role": "assistant",
    "content": " I help",
    "id": "msg_001"
  }
}

data: {
  "id": "chatcmpl-123",
  "delta": {
    "role": "assistant",
    "content": " you today?",
    "id": "msg_001"
  }
}
`,
  out: [
    { type: 'text-start', id: 'msg_001' },
    {
      type: 'text-delta',
      id: 'msg_001',
      delta: 'Hello',
    },
    {
      type: 'text-delta',
      id: 'msg_001',
      delta: ' there',
    },
    {
      type: 'text-delta',
      id: 'msg_001',
      delta: '! How can',
    },
    {
      type: 'text-delta',
      id: 'msg_001',
      delta: ' I help',
    },
    {
      type: 'text-delta',
      id: 'msg_001',
      delta: ' you today?',
    },
    { type: 'text-end', id: 'msg_001' },
  ],
}

/**
 * Chat Agent output with tool calls in structured JSON format
 */
export const CHAT_AGENT_WITH_TOOL_CALLS: LLMOutputFixtures = {
  in: `
data: {
  "id": "chatcmpl-456",
  "delta": {
    "role": "assistant",
    "content": "I'll check the weather for you.",
    "id": "msg_002"
  }
}

data: {
  "id": "chatcmpl-456",
  "delta": {
    "role": "assistant",
    "content": "",
    "id": "msg_002",
    "tool_calls": [
      {
        "type": "function",
        "id": "call_abc123",
        "function": {
          "name": "get_weather",
          "arguments": "{\\"location\\": \\"San Francisco\\", \\"unit\\": \\"celsius\\"}"
        }
      }
    ]
  }
}

data: {
  "id": "chatcmpl-456",
  "delta": {
    "role": "tool",
    "name": "get_weather",
    "content": "{\\"temperature\\": 18, \\"condition\\": \\"partly cloudy\\"}",
    "tool_call_id": "call_abc123",
    "id": "tool_msg_001"
  }
}

data: {
  "id": "chatcmpl-456",
  "delta": {
    "role": "assistant",
    "content": "The weather in San Francisco is 18°C and partly cloudy.",
    "id": "msg_003"
  }
}
`,
  out: [
    { type: 'text-start', id: 'msg_002' },
    {
      type: 'text-delta',
      id: 'msg_002',
      delta: "I'll check the weather for you.",
    },
    { type: 'text-end', id: 'msg_002' },
    {
      type: 'tool-call',
      toolCallId: 'call_abc123',
      toolName: 'get_weather',
      input: '{"location": "San Francisco", "unit": "celsius"}',
      dynamic: true,
      providerExecuted: true,
    },
    {
      type: 'tool-result',
      toolCallId: 'call_abc123',
      toolName: 'get_weather',
      result: '{"temperature": 18, "condition": "partly cloudy"}',
    },
    { type: 'text-start', id: 'msg_003' },
    {
      type: 'text-delta',
      id: 'msg_003',
      delta: 'The weather in San Francisco is 18°C and partly cloudy.',
    },
    { type: 'text-end', id: 'msg_003' },
  ],
}

/**
 * Chat Agent with multiple tool calls
 */
export const CHAT_AGENT_MULTI_TOOL_CALLS: LLMOutputFixtures = {
  in: `
data: {
  "id": "chatcmpl-789",
  "delta": {
    "role": "assistant",
    "content": "Let me fetch both the weather and time.",
    "id": "msg_004"
  }
}

data: {
  "id": "chatcmpl-789",
  "delta": {
    "role": "assistant",
    "content": "",
    "id": "msg_004",
    "tool_calls": [
      {
        "type": "function",
        "id": "call_weather_123",
        "function": {
          "name": "get_weather",
          "arguments": "{\\"location\\": \\"Tokyo\\"}"
        }
      },
      {
        "type": "function",
        "id": "call_time_456",
        "function": {
          "name": "get_current_time",
          "arguments": "{\\"timezone\\": \\"Asia/Tokyo\\"}"
        }
      }
    ]
  }
}
`,
  out: [
    { type: 'text-start', id: 'msg_004' },
    {
      type: 'text-delta',
      id: 'msg_004',
      delta: 'Let me fetch both the weather and time.',
    },
    { type: 'text-end', id: 'msg_004' },
    {
      type: 'tool-call',
      toolCallId: 'call_weather_123',
      toolName: 'get_weather',
      input: '{"location": "Tokyo"}',
      dynamic: true,
      providerExecuted: true,
    },
    {
      type: 'tool-call',
      toolCallId: 'call_time_456',
      toolName: 'get_current_time',
      input: '{"timezone": "Asia/Tokyo"}',
      dynamic: true,
      providerExecuted: true,
    },
  ],
}
