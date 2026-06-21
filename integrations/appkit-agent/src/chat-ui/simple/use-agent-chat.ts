import { useCallback, useMemo, useState } from "react";
import type {
  AssistantPart,
  ChatMessage,
  SSEEvent,
  UseAgentChatOptions,
  UseAgentChatReturn,
} from "./types";
import { serializeForApi } from "./utils";

/**
 * Manages agent chat state and streaming via POST /invocations (Responses API SSE).
 * Returns messages, loading state, input state, submit handler, and derived display list.
 */
export function useAgentChat(
  options: UseAgentChatOptions = {},
): UseAgentChatReturn {
  const { invokeUrl = "/invocations" } = options;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingParts, setStreamingParts] = useState<AssistantPart[]>([]);
  const [streamingText, setStreamingText] = useState("");
  const [loading, setLoading] = useState(false);
  const [input, setInput] = useState("");

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const text = input.trim();
      if (!text || loading) return;

      const userMessage: ChatMessage = { role: "user", content: text };
      setInput("");
      setMessages((prev) => [...prev, userMessage]);
      setLoading(true);
      setStreamingParts([]);
      setStreamingText("");

      try {
        const response = await fetch(invokeUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input: [...messages, userMessage].map(serializeForApi),
            stream: true,
          }),
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(
            (err as { error?: string }).error ?? `HTTP ${response.status}`,
          );
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";
        let fullText = "";
        const parts: AssistantPart[] = [];
        const seenItemIds = new Set<string>();

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6);
            if (payload === "[DONE]") continue;

            let data: SSEEvent;
            try {
              data = JSON.parse(payload);
            } catch {
              continue;
            }

            if (data.type === "response.output_item.added" && data.item) {
              const item = data.item;
              const id =
                item.id ?? `${item.type}_${Date.now()}_${Math.random()}`;
              if (seenItemIds.has(id)) continue;
              seenItemIds.add(id);

              if (item.type === "function_call" && item.name != null) {
                parts.push({
                  type: "function_call",
                  id,
                  callId: item.call_id ?? id,
                  name: item.name,
                  arguments: item.arguments ?? "{}",
                });
              } else if (
                item.type === "function_call_output" &&
                item.call_id != null
              ) {
                parts.push({
                  type: "function_call_output",
                  id,
                  callId: item.call_id,
                  output: item.output ?? "",
                });
              }
              setStreamingParts([...parts]);
            }

            if (data.type === "response.output_text.delta" && data.delta) {
              fullText += data.delta;
              setStreamingText(fullText);
            }
          }
        }

        const finalParts: AssistantPart[] = [...parts];
        if (fullText) {
          finalParts.push({ type: "text", content: fullText });
        }

        setMessages((prev) => [
          ...prev,
          { role: "assistant", parts: finalParts },
        ]);
        setStreamingParts([]);
        setStreamingText("");
      } catch (err) {
        const errorText =
          err instanceof Error ? err.message : "Something went wrong";
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            parts: [{ type: "text", content: `Error: ${errorText}` }],
          },
        ]);
        setStreamingParts([]);
        setStreamingText("");
      } finally {
        setLoading(false);
      }
    },
    [input, loading, messages, invokeUrl],
  );

  const displayMessages = useMemo<ChatMessage[]>(() => {
    if (loading && (streamingParts.length > 0 || streamingText)) {
      const streamingPartList: AssistantPart[] = [...streamingParts];
      if (streamingText) {
        streamingPartList.push({ type: "text", content: streamingText });
      }
      return [
        ...messages,
        { role: "assistant" as const, parts: streamingPartList },
      ];
    }
    return messages;
  }, [messages, streamingParts, streamingText, loading]);

  const isStreamingText = Boolean(loading && streamingText);

  return {
    messages,
    loading,
    input,
    setInput,
    handleSubmit,
    displayMessages,
    isStreamingText,
  };
}
