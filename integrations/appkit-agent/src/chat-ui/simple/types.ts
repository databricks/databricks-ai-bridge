export type TextPart = { type: "text"; content: string };
export type FunctionCallPart = {
  type: "function_call";
  id: string;
  callId: string;
  name: string;
  arguments: string;
};
export type FunctionCallOutputPart = {
  type: "function_call_output";
  id: string;
  callId: string;
  output: string;
};
export type AssistantPart =
  | TextPart
  | FunctionCallPart
  | FunctionCallOutputPart;

export type ChatMessage =
  | { role: "user"; content: string }
  | { role: "assistant"; parts: AssistantPart[] };

export interface SSEItem {
  type?: string;
  id?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  output?: string;
}

export interface SSEEvent {
  type?: string;
  delta?: string;
  error?: string;
  item?: SSEItem;
}

export interface UseAgentChatOptions {
  /** POST URL for invocations (Responses API). Default: "/invocations" */
  invokeUrl?: string;
}

export interface UseAgentChatReturn {
  messages: ChatMessage[];
  loading: boolean;
  input: string;
  setInput: (value: string) => void;
  handleSubmit: (e: React.FormEvent) => void;
  /** Messages + current streaming state for display */
  displayMessages: ChatMessage[];
  /** True when the last message is still streaming text */
  isStreamingText: boolean;
}

export interface SimpleAgentChatProps {
  /** POST URL for invocations. Default: "/invocations" */
  invokeUrl?: string;
  /** Placeholder for the message input */
  placeholder?: string;
  /** Empty state text when there are no messages */
  emptyMessage?: string;
  /** Additional CSS class for the root container */
  className?: string;
}
