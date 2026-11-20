import { AgentChatPart } from "./agent-chat-part";
import type { ChatMessage } from "./types";

interface AgentChatMessageProps {
  message: ChatMessage;
  isLast?: boolean;
  isStreaming?: boolean;
}

/** Renders a single chat message bubble (user or assistant with parts). */
export function AgentChatMessage({
  message,
  isLast = false,
  isStreaming = false,
}: AgentChatMessageProps) {
  if (message.role === "user") {
    return (
      <div className="text-right">
        <span className="text-xs font-medium text-muted-foreground block mb-1">
          You
        </span>
        <div className="inline-block rounded-lg bg-primary text-primary-foreground px-3 py-2 text-sm max-w-[85%]">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="text-left text-foreground space-y-2">
      <span className="text-xs font-medium text-muted-foreground block mb-1">
        Agent
      </span>
      <div className="space-y-2 max-w-[85%]">
        {message.parts.map((part, j) => (
          <AgentChatPart
            key={`part-${part.type}-${j}`}
            part={part}
            showCursor={isLast && j === message.parts.length - 1 && isStreaming}
          />
        ))}
      </div>
    </div>
  );
}
