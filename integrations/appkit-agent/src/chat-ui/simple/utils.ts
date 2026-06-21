import type { ChatMessage } from "./types";

/** Serialize chat message for the Responses API request body. */
export function serializeForApi(msg: ChatMessage): Record<string, unknown> {
  if (msg.role === "user") {
    return { role: "user", content: msg.content };
  }
  const content = msg.parts.map((p) => {
    if (p.type === "text")
      return { type: "output_text" as const, text: p.content };
    if (p.type === "function_call")
      return {
        type: "function_call" as const,
        name: p.name,
        arguments: p.arguments,
      };
    return {
      type: "function_call_output" as const,
      call_id: p.callId,
      output: p.output,
    };
  });
  return { role: "assistant", content };
}

export function tryFormatJson(s: string): string {
  try {
    return JSON.stringify(JSON.parse(s), null, 2);
  } catch {
    return s;
  }
}
