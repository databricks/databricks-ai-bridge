import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useRef, useEffect, useMemo } from "react";

const chatTransport = new DefaultChatTransport({
  api: "/api/chat",
  prepareSendMessagesRequest({ id, messages }) {
    const lastMessage = messages.at(-1);
    const userMessage =
      lastMessage?.role === "user" ? lastMessage : undefined;
    const previousMessages = userMessage ? messages.slice(0, -1) : messages;

    return {
      body: {
        id,
        message: userMessage
          ? { id: userMessage.id, role: "user", parts: userMessage.parts }
          : undefined,
        selectedChatModel: "chat-model",
        selectedVisibilityType: "private",
        previousMessages: previousMessages.map((m) => ({
          id: m.id,
          role: m.role,
          parts: m.parts,
        })),
      },
    };
  },
});

export function AppChat() {
  const { messages, sendMessage, status } = useChat({
    transport: chatTransport,
  });

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const isLoading = status === "streaming" || status === "submitted";

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = inputRef.current?.value.trim();
    if (!text || isLoading) return;
    sendMessage({ text });
    inputRef.current!.value = "";
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Chat</h2>

      <div ref={scrollRef} style={styles.messageList}>
        {messages.length === 0 && (
          <p style={styles.empty}>Send a message to start chatting.</p>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {status === "submitted" && (
          <div style={{ ...styles.bubble, ...styles.assistant }}>
            <Dots />
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          ref={inputRef}
          type="text"
          placeholder="Type a message…"
          disabled={isLoading}
          style={styles.input}
        />
        <button type="submit" disabled={isLoading} style={styles.button}>
          Send
        </button>
      </form>
    </div>
  );
}

function MessageBubble({
  message,
}: {
  message: {
    role: string;
    parts: Array<{ type: string; text?: string }>;
  };
}) {
  const isUser = message.role === "user";
  const text = useMemo(
    () =>
      message.parts
        .filter((p): p is { type: "text"; text: string } => p.type === "text")
        .map((p) => p.text)
        .join(""),
    [message.parts],
  );

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
      }}
    >
      <div
        style={{
          ...styles.bubble,
          ...(isUser ? styles.user : styles.assistant),
        }}
      >
        {text || "…"}
      </div>
    </div>
  );
}

function Dots() {
  return <span style={{ opacity: 0.5 }}>●●●</span>;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    width: "100%",
    maxWidth: 640,
    height: "80vh",
    border: "1px solid #e0e0e0",
    borderRadius: 12,
    overflow: "hidden",
    background: "#fff",
  },
  title: {
    margin: 0,
    padding: "12px 16px",
    borderBottom: "1px solid #e0e0e0",
    fontSize: 16,
    fontWeight: 600,
  },
  messageList: {
    flex: 1,
    overflowY: "auto",
    padding: 16,
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  empty: { color: "#999", fontSize: 14, textAlign: "center", marginTop: 32 },
  bubble: {
    maxWidth: "75%",
    padding: "10px 14px",
    borderRadius: 16,
    fontSize: 14,
    lineHeight: 1.5,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  user: {
    background: "#0066ff",
    color: "#fff",
    borderBottomRightRadius: 4,
  },
  assistant: {
    background: "#f0f0f0",
    color: "#1a1a1a",
    borderBottomLeftRadius: 4,
  },
  form: {
    display: "flex",
    gap: 8,
    padding: 12,
    borderTop: "1px solid #e0e0e0",
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    border: "1px solid #ddd",
    borderRadius: 8,
    fontSize: 14,
    outline: "none",
  },
  button: {
    padding: "10px 20px",
    background: "#0066ff",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
  },
};
