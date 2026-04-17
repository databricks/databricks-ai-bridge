import type { AssistantPart } from "./types";
import { tryFormatJson } from "./utils";

interface AgentChatPartProps {
  part: AssistantPart;
  showCursor?: boolean;
}

/** Renders a single assistant part: text, function_call, or function_call_output. */
export function AgentChatPart({
  part,
  showCursor = false,
}: AgentChatPartProps) {
  if (part.type === "text") {
    return (
      <div className="rounded-lg bg-muted px-3 py-2 text-sm whitespace-pre-wrap">
        {part.content}
        {showCursor && <span className="animate-pulse">|</span>}
      </div>
    );
  }

  if (part.type === "function_call") {
    return (
      <div className="rounded-lg border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm">
        <div className="font-medium text-amber-700 dark:text-amber-400 mb-1">
          Tool: {part.name}
        </div>
        <pre className="text-xs overflow-x-auto text-muted-foreground font-mono whitespace-pre-wrap break-words">
          {tryFormatJson(part.arguments)}
        </pre>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-emerald-500/50 bg-emerald-500/10 px-3 py-2 text-sm">
      <div className="font-medium text-emerald-700 dark:text-emerald-400 mb-1">
        Result
      </div>
      <pre className="text-xs overflow-x-auto text-muted-foreground font-mono whitespace-pre-wrap break-words">
        {tryFormatJson(part.output)}
      </pre>
    </div>
  );
}
