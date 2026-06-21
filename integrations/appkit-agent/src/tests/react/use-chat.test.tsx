// @vitest-environment happy-dom
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { createWrapper } from "./test-helpers";

// Capture the fetch function passed to ChatTransport so we can call it directly
let capturedTransportOptions: Record<string, unknown> = {};
vi.mock("../../react/lib/transport", () => ({
  ChatTransport: class MockChatTransport {
    constructor(opts: Record<string, unknown>) {
      capturedTransportOptions = opts;
    }
  },
}));

const mockFetchWithErrorHandlers = vi
  .fn()
  .mockResolvedValue(new Response());
vi.mock("../../react/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../react/lib/utils")
  >();
  return {
    ...actual,
    fetchWithErrorHandlers: (
      ...args: Parameters<typeof actual.fetchWithErrorHandlers>
    ) => mockFetchWithErrorHandlers(...args),
  };
});

vi.mock("@ai-sdk/react", () => ({
  useChat: vi.fn(() => ({
    messages: [],
    status: "ready",
    sendMessage: vi.fn(),
    setMessages: vi.fn(),
    addToolApprovalResponse: vi.fn(),
    regenerate: vi.fn(),
    resumeStream: vi.fn(),
    clearError: vi.fn(),
    error: null,
  })),
}));

// Must import after mocks are declared (vi.mock is hoisted, but the
// import of the module under test should follow for clarity).
import { useChat } from "../../react/hooks/use-chat";

describe("useChat", () => {
  beforeEach(() => {
    capturedTransportOptions = {};
    mockFetchWithErrorHandlers.mockClear();
  });

  it("stop() recreates AbortController so subsequent fetches use a fresh signal", async () => {
    const { result } = renderHook(() => useChat(), {
      wrapper: createWrapper(),
    });

    const transportFetch = capturedTransportOptions.fetch as (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => Promise<Response>;
    expect(transportFetch).toBeDefined();

    // Fetch before stop — signal should not be aborted
    await transportFetch("http://test", {});
    const firstInit = mockFetchWithErrorHandlers.mock.calls[0][1] as {
      signal: AbortSignal;
    };
    expect(firstInit.signal.aborted).toBe(false);
    const firstSignal = firstInit.signal;

    // Stop the stream
    await act(async () => {
      await result.current.stop();
    });

    // The original signal should now be aborted
    expect(firstSignal.aborted).toBe(true);

    // Fetch after stop — should use a new, non-aborted signal
    mockFetchWithErrorHandlers.mockClear();
    await transportFetch("http://test", {});
    const secondInit = mockFetchWithErrorHandlers.mock.calls[0][1] as {
      signal: AbortSignal;
    };
    expect(secondInit.signal.aborted).toBe(false);
    expect(secondInit.signal).not.toBe(firstSignal);
  });

  it("returns provided id", () => {
    const { result } = renderHook(() => useChat({ id: "my-chat" }), {
      wrapper: createWrapper(),
    });
    expect(result.current.id).toBe("my-chat");
  });

  it("generates a UUID when no id provided", () => {
    const { result } = renderHook(() => useChat(), {
      wrapper: createWrapper(),
    });
    expect(result.current.id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    );
  });

  it("uses external title when provided", () => {
    const { result } = renderHook(
      () => useChat({ title: "External Title" }),
      { wrapper: createWrapper() },
    );
    expect(result.current.title).toBe("External Title");
  });

  it("respects isReadonly option", () => {
    const { result } = renderHook(() => useChat({ isReadonly: true }), {
      wrapper: createWrapper(),
    });
    expect(result.current.isReadonly).toBe(true);
  });

  it("defaults isReadonly to false", () => {
    const { result } = renderHook(() => useChat(), {
      wrapper: createWrapper(),
    });
    expect(result.current.isReadonly).toBe(false);
  });

  it("passes feedback through", () => {
    const feedback = {
      "msg-1": {
        messageId: "msg-1",
        feedbackType: "thumbs_up" as const,
        assessmentId: null,
      },
    };
    const { result } = renderHook(() => useChat({ feedback }), {
      wrapper: createWrapper(),
    });
    expect(result.current.feedback).toBe(feedback);
  });
});
