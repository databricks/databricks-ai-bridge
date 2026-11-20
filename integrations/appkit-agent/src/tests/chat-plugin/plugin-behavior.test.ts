import { beforeEach, describe, expect, test, vi } from "vitest";
import {
  createMockRequest,
  createMockResponse,
  createMockRouter,
} from "./test-helpers";

const {
  ensureSchemaMock,
  createDbMock,
  getChatByIdMock,
  getMessagesByChatIdMock,
  saveMessagesMock,
  saveChatMock,
  updateChatTitleByIdMock,
  updateChatLastContextByIdMock,
  languageModelMock,
  createChatProviderMock,
  drainStreamToWriterMock,
  fallbackToGenerateTextMock,
  streamTextMock,
  convertToModelMessagesMock,
  generateTextMock,
  pipeUIMessageStreamToResponseMock,
  pendingExecutions,
} = vi.hoisted(() => ({
  ensureSchemaMock: vi.fn(),
  createDbMock: vi.fn(() => ({})),
  getChatByIdMock: vi.fn(async () => null),
  getMessagesByChatIdMock: vi.fn(async () => []),
  saveMessagesMock: vi.fn(async () => undefined),
  saveChatMock: vi.fn(async () => undefined),
  updateChatTitleByIdMock: vi.fn(async () => undefined),
  updateChatLastContextByIdMock: vi.fn(async () => undefined),
  languageModelMock: vi.fn(async () => ({})),
  createChatProviderMock: vi.fn(() => ({
    languageModel: languageModelMock,
  })),
  drainStreamToWriterMock: vi.fn(async () => ({ failed: false })),
  fallbackToGenerateTextMock: vi.fn<
    () => Promise<
      | {
          usage: { inputTokens: number; outputTokens: number };
          traceId?: string;
        }
      | undefined
    >
  >(async () => undefined),
  streamTextMock: vi.fn(() => ({
    toUIMessageStream: vi.fn(() => ({})),
  })),
  convertToModelMessagesMock: vi.fn(async () => []),
  generateTextMock: vi.fn(async () => ({ text: "generated title" })),
  pipeUIMessageStreamToResponseMock: vi.fn(),
  pendingExecutions: [] as Promise<unknown>[],
}));

vi.mock("@databricks/appkit", () => {
  class MockPlugin<T> {
    protected config: T;
    constructor(config: T) {
      this.config = config;
    }
    registerEndpoint(_name: string, _path: string) {}
  }

  return {
    Plugin: MockPlugin,
    toPlugin: (Ctor: new (config: unknown) => unknown) => (config: unknown) => {
      const instance = new Ctor(config) as Record<string, unknown>;
      return {
        name: instance.name,
        instance,
      };
    },
  };
});

vi.mock("../../chat-plugin/migrate", () => ({
  ensureSchema: ensureSchemaMock,
}));

vi.mock("../../chat-plugin/provider", () => ({
  CONTEXT_HEADER_CONVERSATION_ID: "x-databricks-conversation-id",
  CONTEXT_HEADER_USER_ID: "x-databricks-user-id",
  createChatProvider: createChatProviderMock,
}));

vi.mock("../../chat-plugin/persistence", () => ({
  createDb: createDbMock,
  getChatById: getChatByIdMock,
  getMessagesByChatId: getMessagesByChatIdMock,
  saveMessages: saveMessagesMock,
  saveChat: saveChatMock,
  updateChatTitleById: updateChatTitleByIdMock,
  updateChatLastContextById: updateChatLastContextByIdMock,
  getChatsByUserId: vi.fn(),
  getMessageById: vi.fn(),
  deleteMessagesAfter: vi.fn(),
  getVotesByChatId: vi.fn(),
  voteMessage: vi.fn(),
  updateChatVisibilityById: vi.fn(),
  deleteChatById: vi.fn(),
}));

vi.mock("../../chat-plugin/stream-fallback", () => ({
  drainStreamToWriter: drainStreamToWriterMock,
  fallbackToGenerateText: fallbackToGenerateTextMock,
}));

vi.mock("ai", () => ({
  convertToModelMessages: convertToModelMessagesMock,
  createUIMessageStream: vi.fn(
    ({
      execute,
      onFinish,
    }: {
      execute: (args: {
        writer: {
          write: (value: unknown) => void;
          onError?: (e: unknown) => void;
        };
      }) => Promise<void>;
      onFinish?: (args: {
        responseMessage: { id: string; role: "assistant"; parts: unknown[] };
      }) => Promise<void>;
    }) => {
      const writer = {
        write: vi.fn(),
        onError: vi.fn(),
      };
      const run = execute({ writer }).then(async () => {
        if (onFinish) {
          await onFinish({
            responseMessage: {
              id: "assistant-msg-1",
              role: "assistant",
              parts: [],
            },
          });
        }
      });
      pendingExecutions.push(run);
      return { kind: "mock-stream" };
    },
  ),
  generateText: generateTextMock,
  pipeUIMessageStreamToResponse: pipeUIMessageStreamToResponseMock,
  streamText: streamTextMock,
}));

import { ChatPlugin } from "../../chat-plugin/index";

describe("ChatPlugin behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    pendingExecutions.length = 0;
    ensureSchemaMock.mockResolvedValue(undefined);
    drainStreamToWriterMock.mockResolvedValue({ failed: false });
    fallbackToGenerateTextMock.mockResolvedValue(undefined);
    pipeUIMessageStreamToResponseMock.mockImplementation(() => undefined);
  });

  test("runs auto-migration by default when pool is configured", async () => {
    const plugin = new ChatPlugin({ pool: {} as never });
    await plugin.setup();

    expect(ensureSchemaMock).toHaveBeenCalledTimes(1);
  });

  test("skips auto-migration when explicitly disabled", async () => {
    const plugin = new ChatPlugin({
      pool: {} as never,
      autoMigrate: false,
    });
    await plugin.setup();

    expect(ensureSchemaMock).not.toHaveBeenCalled();
  });

  test("uses local plugin backend proxy in main chat handler", async () => {
    const plugin = new ChatPlugin({ backend: "agent" });
    await plugin.setup();
    createChatProviderMock.mockClear();
    languageModelMock.mockClear();
    const { router, getHandler } = createMockRouter() as {
      router: Record<string, unknown>;
      getHandler: (method: string, path: string) => (req: unknown, res: unknown) => Promise<void>;
    };
    plugin.injectRoutes(router as never);

    const req = createMockRequest({
      protocol: "https",
      get: vi.fn().mockReturnValue("example.local"),
      body: {
        id: "chat-local-backend",
        message: {
          id: "msg-1",
          role: "user",
          parts: [{ type: "text", text: "hello" }],
        },
        selectedChatModel: "chat-model",
        selectedVisibilityType: "private",
      },
      session: { user: { id: "user-1", email: "u@databricks.com" } },
    });
    const res = createMockResponse();
    const handler = getHandler("POST", "/");

    await handler(req, res);
    await Promise.allSettled(pendingExecutions);

    expect(createChatProviderMock).toHaveBeenCalledWith({
      apiProxy: "https://example.local/api/agent",
    });
    expect(languageModelMock).toHaveBeenCalledWith("chat-model");
  });

  test("uses explicit backend proxy in main chat handler", async () => {
    const plugin = new ChatPlugin({
      backend: { proxy: "http://localhost:9000/invocations" },
    });
    await plugin.setup();
    createChatProviderMock.mockClear();
    languageModelMock.mockClear();
    const { router, getHandler } = createMockRouter() as {
      router: Record<string, unknown>;
      getHandler: (method: string, path: string) => (req: unknown, res: unknown) => Promise<void>;
    };
    plugin.injectRoutes(router as never);

    const req = createMockRequest({
      body: {
        id: "chat-proxy-backend",
        message: {
          id: "msg-2",
          role: "user",
          parts: [{ type: "text", text: "hello via proxy" }],
        },
        selectedChatModel: "chat-model",
        selectedVisibilityType: "private",
      },
      session: { user: { id: "user-2", email: "u2@databricks.com" } },
    });
    const res = createMockResponse();
    const handler = getHandler("POST", "/");

    await handler(req, res);
    await Promise.allSettled(pendingExecutions);

    expect(createChatProviderMock).toHaveBeenCalledWith({
      apiProxy: "http://localhost:9000/invocations",
    });
    expect(languageModelMock).toHaveBeenCalledWith("chat-model");
  });

  test("uses backend endpoint override in main chat handler", async () => {
    const plugin = new ChatPlugin({
      backend: { endpoint: "databricks-claude-sonnet-4-5" },
    });
    await plugin.setup();
    createChatProviderMock.mockClear();
    languageModelMock.mockClear();
    const { router, getHandler } = createMockRouter() as {
      router: Record<string, unknown>;
      getHandler: (method: string, path: string) => (req: unknown, res: unknown) => Promise<void>;
    };
    plugin.injectRoutes(router as never);

    const req = createMockRequest({
      body: {
        id: "chat-endpoint-backend",
        message: {
          id: "msg-3",
          role: "user",
          parts: [{ type: "text", text: "hello endpoint" }],
        },
        selectedChatModel: "chat-model",
        selectedVisibilityType: "private",
      },
      session: { user: { id: "user-3", email: "u3@databricks.com" } },
    });
    const res = createMockResponse();
    const handler = getHandler("POST", "/");

    await handler(req, res);
    await Promise.allSettled(pendingExecutions);

    expect(createChatProviderMock).not.toHaveBeenCalled();
    expect(languageModelMock).toHaveBeenCalledWith(
      "databricks-claude-sonnet-4-5",
    );
  });

  test("registers expected chat routes", async () => {
    const plugin = new ChatPlugin({ pool: {} as never, autoMigrate: false });
    await plugin.setup();
    const { router } = createMockRouter() as {
      router: Record<string, ReturnType<typeof vi.fn>>;
    };
    plugin.injectRoutes(router as never);

    expect(router.get).toHaveBeenCalledWith("/config", expect.any(Function));
    expect(router.get).toHaveBeenCalledWith("/session", expect.any(Function));
    expect(router.get).toHaveBeenCalledWith(
      "/history",
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.get).toHaveBeenCalledWith(
      "/messages/:id",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.delete).toHaveBeenCalledWith(
      "/messages/:id/trailing",
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.post).toHaveBeenCalledWith(
      "/feedback",
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.get).toHaveBeenCalledWith(
      "/feedback/chat/:chatId",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.post).toHaveBeenCalledWith(
      "/title",
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.patch).toHaveBeenCalledWith(
      "/:id/visibility",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.get).toHaveBeenCalledWith(
      "/:id/stream",
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.get).toHaveBeenCalledWith(
      "/:id",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.delete).toHaveBeenCalledWith(
      "/:id",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
    expect(router.post).toHaveBeenCalledWith(
      "/",
      expect.any(Function),
      expect.any(Function),
    );
  });

  test("ends request early for MCP denial in previous assistant messages", async () => {
    const plugin = new ChatPlugin({ pool: {} as never, autoMigrate: false });
    await plugin.setup();
    const { router, getHandler } = createMockRouter() as {
      router: Record<string, unknown>;
      getHandler: (
        method: string,
        path: string,
      ) => (req: unknown, res: unknown) => Promise<void>;
    };
    plugin.injectRoutes(router as never);

    const handler = getHandler("POST", "/");
    const req = createMockRequest({
      body: {
        id: "chat-1",
        selectedChatModel: "chat-model",
        selectedVisibilityType: "private",
        previousMessages: [
          {
            id: "assistant-1",
            role: "assistant",
            parts: [{ type: "dynamic-tool", state: "output-denied" }],
          },
        ],
      },
      session: { user: { id: "user-1", email: "u@databricks.com" } },
    });
    const res = createMockResponse();

    await handler(req, res);

    expect(res.end).toHaveBeenCalledTimes(1);
    expect(streamTextMock).not.toHaveBeenCalled();
  });

  test("falls back to generateText when stream draining fails", async () => {
    drainStreamToWriterMock.mockResolvedValueOnce({ failed: true });
    fallbackToGenerateTextMock.mockResolvedValueOnce({
      usage: { inputTokens: 1, outputTokens: 2 },
      traceId: "trace-1",
    });

    const plugin = new ChatPlugin({});
    await plugin.setup();
    const { router, getHandler } = createMockRouter() as {
      router: Record<string, unknown>;
      getHandler: (
        method: string,
        path: string,
      ) => (req: unknown, res: unknown) => Promise<void>;
    };
    plugin.injectRoutes(router as never);

    const handler = getHandler("POST", "/");
    const req = createMockRequest({
      body: {
        id: "chat-2",
        message: {
          id: "msg-1",
          role: "user",
          parts: [{ type: "text", text: "hello" }],
        },
        selectedChatModel: "chat-model",
        selectedVisibilityType: "private",
      },
      session: { user: { id: "user-2", email: "u2@databricks.com" } },
    });
    const res = createMockResponse();

    await handler(req, res);
    await Promise.allSettled(pendingExecutions);

    expect(fallbackToGenerateTextMock).toHaveBeenCalledTimes(1);
  });
});
