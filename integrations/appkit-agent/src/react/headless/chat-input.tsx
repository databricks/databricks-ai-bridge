import {
  useState,
  useCallback,
  type ReactNode,
  type FormEvent,
} from "react";
import type { ChatStatus } from "ai";
import type { ChatMessage } from "../types.js";
import type { UseChatHelpers } from "@ai-sdk/react";

export interface ChatInputRenderProps {
  value: string;
  onChange: (value: string) => void;
  submit: (e?: FormEvent) => void;
  isStreaming: boolean;
  stop: () => void;
  canSubmit: boolean;
  handleKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>) => void;
}

export interface ChatInputProps {
  onSubmit: UseChatHelpers<ChatMessage>["sendMessage"];
  status: ChatStatus;
  onStop: () => void;
  children: (props: ChatInputRenderProps) => ReactNode;
}

export function ChatInput({
  onSubmit,
  status,
  onStop,
  children,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const isStreaming = status === "streaming";

  const submit = useCallback(
    (e?: FormEvent) => {
      e?.preventDefault();
      const trimmed = value.trim();
      if (!trimmed) return;

      onSubmit({
        role: "user" as const,
        parts: [{ type: "text" as const, text: trimmed }],
      });
      setValue("");
    },
    [value, onSubmit],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>) => {
      if (e.key === "Enter") {
        if (e.nativeEvent.isComposing) return;
        if (e.shiftKey) return;
        e.preventDefault();
        submit();
      }
    },
    [submit],
  );

  return children({
    value,
    onChange: setValue,
    submit,
    isStreaming,
    stop: onStop,
    canSubmit: value.trim().length > 0 && !isStreaming,
    handleKeyDown,
  });
}
