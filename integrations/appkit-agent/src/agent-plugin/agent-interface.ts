/**
 * Agent interface types for the AppKit Agent Plugin.
 *
 * These types define the contract between the plugin framework and agent
 * implementations. They mirror the OpenAI Responses API SSE format without
 * requiring the `openai` package as a dependency.
 */

// ---------------------------------------------------------------------------
// Invoke params
// ---------------------------------------------------------------------------

export interface InvokeParams {
  input: string;
  chat_history?: Array<{ role: string; content: string }>;
}

// ---------------------------------------------------------------------------
// Responses API output types (minimal subset)
// ---------------------------------------------------------------------------

export interface ResponseOutputTextContent {
  type: "output_text";
  text: string;
  annotations: unknown[];
}

export interface ResponseOutputMessage {
  id: string;
  type: "message";
  role: "assistant";
  status: "in_progress" | "completed";
  content: ResponseOutputTextContent[];
}

export interface ResponseFunctionToolCall {
  id: string;
  type: "function_call";
  call_id: string;
  name: string;
  arguments: string;
  status: "completed";
}

export interface ResponseFunctionCallOutput {
  id: string;
  type: "function_call_output";
  call_id: string;
  output: string;
}

export type ResponseOutputItem =
  | ResponseOutputMessage
  | ResponseFunctionToolCall
  | ResponseFunctionCallOutput;

// ---------------------------------------------------------------------------
// Responses API SSE event types
// ---------------------------------------------------------------------------

export interface ResponseOutputItemAddedEvent {
  type: "response.output_item.added";
  item: ResponseOutputItem;
  output_index: number;
  sequence_number: number;
}

export interface ResponseOutputItemDoneEvent {
  type: "response.output_item.done";
  item: ResponseOutputItem;
  output_index: number;
  sequence_number: number;
}

export interface ResponseTextDeltaEvent {
  type: "response.output_text.delta";
  item_id: string;
  output_index: number;
  content_index: number;
  delta: string;
  sequence_number: number;
}

export interface ResponseCompletedEvent {
  type: "response.completed";
  sequence_number: number;
  response: Record<string, unknown>;
}

export interface ResponseErrorEvent {
  type: "error";
  error: string;
}

export interface ResponseFailedEvent {
  type: "response.failed";
}

export type ResponseStreamEvent =
  | ResponseOutputItemAddedEvent
  | ResponseOutputItemDoneEvent
  | ResponseTextDeltaEvent
  | ResponseCompletedEvent
  | ResponseErrorEvent
  | ResponseFailedEvent;

// ---------------------------------------------------------------------------
// Agent interface
// ---------------------------------------------------------------------------

/**
 * Contract that agent implementations must fulfil.
 *
 * The plugin calls `invoke()` for non-streaming requests and `stream()` for
 * SSE streaming. Implementations are responsible for translating their SDK's
 * output into Responses API types.
 */
export interface AgentInterface {
  invoke(params: InvokeParams): Promise<ResponseOutputItem[]>;
  stream(params: InvokeParams): AsyncGenerator<ResponseStreamEvent>;
}
