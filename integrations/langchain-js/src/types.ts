/**
 * OpenAI-compatible types for Databricks Model Serving
 *
 * These types match the OpenAI Chat Completions API which Databricks Model Serving implements.
 */

// ============================================================================
// Request Types
// ============================================================================

export type ChatCompletionRole = "system" | "user" | "assistant" | "tool";

export interface ChatCompletionMessageToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatCompletionSystemMessageParam {
  role: "system";
  content: string;
  name?: string;
}

export interface ChatCompletionUserMessageParam {
  role: "user";
  content: string | ChatCompletionContentPart[];
  name?: string;
}

export interface ChatCompletionContentPartText {
  type: "text";
  text: string;
}

export interface ChatCompletionContentPartImage {
  type: "image_url";
  image_url: {
    url: string;
    detail?: "auto" | "low" | "high";
  };
}

export type ChatCompletionContentPart =
  | ChatCompletionContentPartText
  | ChatCompletionContentPartImage;

export interface ChatCompletionAssistantMessageParam {
  role: "assistant";
  content?: string | null;
  name?: string;
  tool_calls?: ChatCompletionMessageToolCall[];
}

export interface ChatCompletionToolMessageParam {
  role: "tool";
  content: string;
  tool_call_id: string;
}

export type ChatCompletionMessageParam =
  | ChatCompletionSystemMessageParam
  | ChatCompletionUserMessageParam
  | ChatCompletionAssistantMessageParam
  | ChatCompletionToolMessageParam;

export interface ChatCompletionTool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
    strict?: boolean;
  };
}

export type ChatCompletionToolChoiceOption =
  | "none"
  | "auto"
  | "required"
  | {
      type: "function";
      function: {
        name: string;
      };
    };

export interface ChatCompletionRequest {
  messages: ChatCompletionMessageParam[];
  model?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  stream?: boolean;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  tools?: ChatCompletionTool[];
  tool_choice?: ChatCompletionToolChoiceOption;
  response_format?: { type: "text" | "json_object" };
  seed?: number;
  user?: string;
}

// ============================================================================
// Response Types
// ============================================================================

export interface ChatCompletionChoice {
  index: number;
  message: {
    role: "assistant";
    content: string | null;
    tool_calls?: ChatCompletionMessageToolCall[];
  };
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
  logprobs?: null;
}

export interface CompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage?: CompletionUsage;
  system_fingerprint?: string;
}

// ============================================================================
// Streaming Types
// ============================================================================

export interface ChatCompletionChunkChoiceDelta {
  role?: "assistant";
  content?: string | null;
  tool_calls?: ChatCompletionChunkChoiceDeltaToolCall[];
}

export interface ChatCompletionChunkChoiceDeltaToolCall {
  index: number;
  id?: string;
  type?: "function";
  function?: {
    name?: string;
    arguments?: string;
  };
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: ChatCompletionChunkChoiceDelta;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
  logprobs?: null;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
  usage?: CompletionUsage;
  system_fingerprint?: string;
}

// ============================================================================
// Error Types
// ============================================================================

export interface DatabricksApiError {
  error?: {
    message: string;
    type?: string;
    code?: string;
  };
  message?: string;
  error_code?: string;
}
