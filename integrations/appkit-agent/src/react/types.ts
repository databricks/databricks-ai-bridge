import type { LanguageModelUsage, UIMessage, UIMessagePart } from "ai";

// ── Custom data types for the AI SDK stream ─────────────────────────────────

export type CustomUIDataTypes = {
  error: string;
  usage: LanguageModelUsage;
  traceId: string | null;
  title: string;
};

type MessageMetadata = {
  createdAt: string;
};

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export type ChatTools = {};

export type ChatMessage = UIMessage<MessageMetadata, CustomUIDataTypes>;

// ── Domain types ─────────────────────────────────────────────────────────────

export interface Attachment {
  name: string;
  url: string;
  contentType: string;
}

export type VisibilityType = "private" | "public";

export interface Feedback {
  messageId: string;
  feedbackType: "thumbs_up" | "thumbs_down";
  assessmentId: string | null;
}

export type FeedbackMap = Record<string, Feedback>;

export interface ClientSession {
  user: {
    email: string;
    name?: string;
    preferredUsername?: string;
  } | null;
}

export interface Chat {
  id: string;
  createdAt: Date;
  title: string;
  userId: string;
  visibility: "public" | "private";
  lastContext: {
    inputTokens?: {
      total?: number;
      noCache?: number;
      cacheRead?: number;
      cacheWrite?: number;
    };
    outputTokens?: {
      total?: number;
      text?: number;
      reasoning?: number;
    };
  } | null;
}

export interface DBMessage {
  id: string;
  chatId: string;
  role: string;
  parts: unknown;
  attachments: unknown;
  createdAt: Date;
  traceId: string | null;
}

// ── Errors ───────────────────────────────────────────────────────────────────

type ErrorType =
  | "bad_request"
  | "unauthorized"
  | "forbidden"
  | "not_found"
  | "rate_limit"
  | "offline"
  | "empty";

type Surface =
  | "chat"
  | "auth"
  | "api"
  | "database"
  | "history"
  | "stream"
  | "message";

export type ErrorCode = `${ErrorType}:${Surface}`;

export class ChatSDKError extends Error {
  public type: ErrorType;
  public surface: Surface;
  public statusCode: number;
  public override cause?: string;

  constructor(errorCode: ErrorCode, cause?: string) {
    super();

    const [_type, _surface] = errorCode.split(":");
    const type = _type as ErrorType;
    const surface = _surface as Surface;

    this.type = type;
    this.cause = cause;
    this.surface = surface;
    this.message = getMessageByErrorCode(errorCode);
    this.statusCode = getStatusCodeByType(this.type);
  }
}

function getStatusCodeByType(type: ErrorType): number {
  switch (type) {
    case "bad_request":
      return 400;
    case "unauthorized":
      return 401;
    case "forbidden":
      return 403;
    case "not_found":
      return 404;
    case "rate_limit":
      return 429;
    case "offline":
      return 0;
    default:
      return 500;
  }
}

export function getMessageByErrorCode(errorCode: ErrorCode): string {
  if (errorCode.includes("database")) {
    return "An error occurred while executing a database query.";
  }

  switch (errorCode) {
    case "bad_request:api":
      return "The request couldn't be processed. Please check your input and try again.";
    case "unauthorized:auth":
      return "You need to sign in before continuing.";
    case "forbidden:auth":
      return "Your account does not have access to this feature.";
    case "rate_limit:chat":
      return "You have exceeded your maximum number of messages for the day. Please try again later.";
    case "not_found:chat":
      return "The requested chat was not found. Please check the chat ID and try again.";
    case "forbidden:chat":
      return "This chat belongs to another user. Please check the chat ID and try again.";
    case "unauthorized:chat":
      return "You need to sign in to view this chat. Please sign in and try again.";
    case "offline:chat":
      return "We're having trouble sending your message. Please check your internet connection and try again.";
    default:
      return "Something went wrong. Please try again later.";
  }
}

// ── Chat features ────────────────────────────────────────────────────────────

export interface ChatFeatures {
  chatHistory: boolean;
  feedback: boolean;
}

// ── Chat history page ────────────────────────────────────────────────────────

export interface ChatHistoryPage {
  chats: Array<Chat>;
  hasMore: boolean;
}

// ── Utility: convert DB messages to UI messages ──────────────────────────────

export function convertToUIMessages(
  messages: DBMessage[],
): ChatMessage[] {
  return messages.map((message) => ({
    id: message.id,
    role: message.role as "user" | "assistant" | "system",
    parts: message.parts as UIMessagePart<CustomUIDataTypes, ChatTools>[],
    metadata: {
      createdAt:
        typeof message.createdAt === "string"
          ? message.createdAt
          : message.createdAt.toISOString(),
    },
  }));
}
