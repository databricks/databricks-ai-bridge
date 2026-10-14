import { useRef, useEffect, useCallback, useState, type ReactNode } from "react";
import { useChat, type UseChatOptions } from "../hooks/use-chat.js";
import type { ChatMessage } from "../types.js";
import type { UseChatHelpers } from "@ai-sdk/react";

export interface ConversationRenderProps {
  messages: ChatMessage[];
  status: UseChatHelpers<ChatMessage>["status"];
  sendMessage: UseChatHelpers<ChatMessage>["sendMessage"];
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  addToolApprovalResponse: UseChatHelpers<ChatMessage>["addToolApprovalResponse"];
  regenerate: UseChatHelpers<ChatMessage>["regenerate"];
  stop: () => void;
  id: string;
  title: string | undefined;
  isTitleLoading: boolean;
  isReadonly: boolean;
  isAtBottom: boolean;
  scrollToBottom: (behavior?: ScrollBehavior) => void;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

export interface ConversationProps extends UseChatOptions {
  children: (props: ConversationRenderProps) => ReactNode;
}

export function Conversation({ children, ...chatOptions }: ConversationProps) {
  const chat = useChat(chatOptions);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const threshold = 50;
    setIsAtBottom(
      el.scrollHeight - el.scrollTop - el.clientHeight < threshold,
    );
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("scroll", handleScroll, { passive: true });
    return () => el.removeEventListener("scroll", handleScroll);
  }, [handleScroll]);

  useEffect(() => {
    if (isAtBottom && containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [chat.messages, isAtBottom]);

  const scrollToBottom = useCallback(
    (behavior: ScrollBehavior = "smooth") => {
      containerRef.current?.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior,
      });
    },
    [],
  );

  return children({
    messages: chat.messages,
    status: chat.status,
    sendMessage: chat.sendMessage,
    setMessages: chat.setMessages,
    addToolApprovalResponse: chat.addToolApprovalResponse,
    regenerate: chat.regenerate,
    stop: chat.stop,
    id: chat.id,
    title: chat.title,
    isTitleLoading: chat.isTitleLoading,
    isReadonly: chat.isReadonly,
    isAtBottom,
    scrollToBottom,
    containerRef,
  });
}
