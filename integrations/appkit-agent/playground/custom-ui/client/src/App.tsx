import { useState } from "react";
import {
  ChatAgentProvider,
  ChatInput,
  useChat,
  useChatData,
  useHistory,
  useSession,
  generateUUID,
  type ChatMessage,
  type FeedbackMap,
} from "@databricks/appkit-agent/react";

export function App() {
  const [activeChatId, setActiveChatId] = useState<string | undefined>();

  return (
    <ChatAgentProvider
      apiBase="/api/chat"
      onNavigate={(chatId) => setActiveChatId(chatId)}
    >
      <div className="wrapper">
        <div className="layout">
          <Sidebar activeChatId={activeChatId} onSelect={setActiveChatId} />
          <ChatView key={activeChatId ?? "new"} chatId={activeChatId} />
        </div>
      </div>
    </ChatAgentProvider>
  );
}

function Sidebar({
  activeChatId,
  onSelect,
}: {
  activeChatId: string | undefined;
  onSelect: (id: string | undefined) => void;
}) {
  const { chats, isLoading, hasMore, loadMore, deleteChat } = useHistory();
  const { user } = useSession();

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <strong>Chats</strong>
        <button onClick={() => onSelect(undefined)} className="new-chat-btn">
          + New
        </button>
      </div>

      {user && <div className="user-info">{user.email}</div>}

      <div className="chat-list">
        {isLoading && chats.length === 0 && (
          <div className="muted">Loading...</div>
        )}

        {chats.map((c) => (
          <div
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={`chat-item${c.id === activeChatId ? " active" : ""}`}
          >
            <span className="chat-item-title">{c.title || "Untitled"}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                deleteChat(c.id);
              }}
              className="chat-item-delete"
            >
              ×
            </button>
          </div>
        ))}

        {hasMore && (
          <button onClick={loadMore} className="load-more-btn">
            Load more
          </button>
        )}
      </div>
    </aside>
  );
}

function ChatView({ chatId }: { chatId: string | undefined }) {
  const isExisting = chatId !== undefined;
  const { chatData, isLoading, error } = useChatData(
    isExisting ? chatId : undefined,
  );

  if (isExisting && isLoading) {
    return (
      <main className="chat-panel">
        <div className="empty-state">Loading chat...</div>
      </main>
    );
  }

  if (isExisting && error) {
    return (
      <main className="chat-panel">
        <div className="empty-state">{error}</div>
      </main>
    );
  }

  return (
    <Chat
      chatId={chatId}
      initialMessages={chatData?.messages}
      title={chatData?.chat.title}
      feedback={chatData?.feedback}
    />
  );
}

function Chat({
  chatId,
  initialMessages,
  title,
  feedback,
}: {
  chatId: string | undefined;
  initialMessages?: ChatMessage[];
  title?: string;
  feedback?: FeedbackMap;
}) {
  const [id] = useState(() => chatId ?? generateUUID());
  const chat = useChat({
    id,
    initialMessages: initialMessages ?? [],
    title,
    feedback: feedback ?? {},
  });

  return (
    <main className="chat-panel">
      <div className="chat-header">
        <span className="chat-title">
          {chat.title || (chat.messages.length === 0 ? "New Chat" : "Chat")}
          {chat.isTitleLoading && " ..."}
        </span>
      </div>

      <div className="message-list">
        {chat.messages.length === 0 && (
          <div className="empty-state">Send a message to start chatting.</div>
        )}

        {chat.messages.map((m) => (
          <div key={m.id} className={`message ${m.role}`}>
            {m.parts.map((p, i) =>
              p.type === "text" ? <span key={i}>{p.text}</span> : null,
            )}
          </div>
        ))}

        {chat.status === "submitted" && (
          <div className="message thinking">Thinking...</div>
        )}
      </div>

      <ChatInput
        onSubmit={chat.sendMessage}
        status={chat.status}
        onStop={chat.stop}
      >
        {({ value, onChange, submit, isStreaming, stop, handleKeyDown }) => (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
            className="input-form"
          >
            <input
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message..."
              className="input-field"
              autoFocus
            />
            {isStreaming ? (
              <button type="button" onClick={stop} className="stop-btn">
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!value.trim()}
                className="send-btn"
              >
                Send
              </button>
            )}
          </form>
        )}
      </ChatInput>
    </main>
  );
}
