import { useEffect, useRef } from "react";
import { cn } from "./primitives/cn";
import { Button } from "./primitives/button";
import { Card } from "./primitives/card";
import { AgentChatMessage } from "./agent-chat-message";
import type { SimpleAgentChatProps, ChatMessage } from "./types";
import { useAgentChat } from "./use-agent-chat";

/** Agent chat UI: message list + input, wired to POST /invocations SSE streaming. */
export function SimpleAgentChat({
  invokeUrl = "/invocations",
  placeholder = "Type a message...",
  emptyMessage = "Send a message to start.",
  className,
}: SimpleAgentChatProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const {
    displayMessages,
    loading,
    input,
    setInput,
    handleSubmit,
    isStreamingText,
  } = useAgentChat({ invokeUrl });

  const contentLength = displayMessages.length;
  // biome-ignore lint/correctness/useExhaustiveDependencies: deps used as triggers for auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [contentLength, isStreamingText]);

  return (
    <div data-chat-ui="" className={cn("flex flex-col min-h-0", className)}>
      <Card className="flex-1 flex flex-col min-h-0 p-4">
        <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 mb-4">
          {displayMessages.length === 0 && (
            <p className="text-muted-foreground text-sm">{emptyMessage}</p>
          )}
          {displayMessages.map((msg, i) => (
            <MessageItem
              key={`msg-${i}-${msg.role}`}
              message={msg}
              isLast={i === displayMessages.length - 1}
              isStreaming={isStreamingText}
            />
          ))}
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={placeholder}
            className="flex-1 rounded-lg border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            disabled={loading}
          />
          <Button type="submit" disabled={loading || !input.trim()}>
            {loading ? "..." : "Send"}
          </Button>
        </form>
      </Card>
    </div>
  );
}

function MessageItem({
  message,
  isLast,
  isStreaming,
}: {
  message: ChatMessage;
  isLast: boolean;
  isStreaming: boolean;
}) {
  return (
    <AgentChatMessage
      message={message}
      isLast={isLast}
      isStreaming={isStreaming}
    />
  );
}
