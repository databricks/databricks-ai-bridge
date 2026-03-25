// @vitest-environment happy-dom
import { describe, it, expect, vi, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useChatData } from "../../react/hooks/use-chat-data";
import { createWrapper } from "./test-helpers";

describe("useChatData", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("returns chat data on successful fetch", async () => {
    const mockChat = { id: "chat-1", title: "Test Chat" };
    const mockMessages = [
      {
        id: "msg-1",
        chatId: "chat-1",
        role: "user",
        parts: [{ type: "text", text: "hello" }],
        attachments: [],
        createdAt: "2026-01-01T00:00:00Z",
        traceId: null,
      },
    ];
    const mockFeedback = {
      "msg-1": {
        messageId: "msg-1",
        feedbackType: "thumbs_up",
        assessmentId: null,
      },
    };

    global.fetch = vi.fn(((url: string) => {
      if (url.includes("/feedback/chat/")) {
        return Promise.resolve(
          new Response(JSON.stringify(mockFeedback), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      if (url.includes("/messages/")) {
        return Promise.resolve(
          new Response(JSON.stringify(mockMessages), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      return Promise.resolve(
        new Response(JSON.stringify(mockChat), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }) as typeof fetch);

    const { result } = renderHook(() => useChatData("chat-1"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.chatData).not.toBeUndefined();
    });

    expect(result.current.chatData!.chat).toEqual(mockChat);
    expect(result.current.chatData!.messages).toHaveLength(1);
    expect(result.current.chatData!.messages[0].parts).toEqual(
      mockMessages[0].parts,
    );
    expect(result.current.chatData!.feedback).toEqual(mockFeedback);
    expect(result.current.error).toBeNull();
  });

  it("returns null data and error string for 404 chat", async () => {
    global.fetch = vi.fn((() =>
      Promise.resolve(new Response(null, { status: 404 }))) as typeof fetch);

    const { result } = renderHook(() => useChatData("missing-chat"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.chatData).toBeNull();
    expect(result.current.error).toBe(
      "Chat not found or you do not have access",
    );
  });

  it("returns null data and error string for 403 chat", async () => {
    global.fetch = vi.fn((() =>
      Promise.resolve(new Response(null, { status: 403 }))) as typeof fetch);

    const { result } = renderHook(() => useChatData("forbidden-chat"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.chatData).toBeNull();
    expect(result.current.error).toBe(
      "Chat not found or you do not have access",
    );
  });

  it("returns error string when fetch throws (e.g. 500)", async () => {
    global.fetch = vi.fn((() =>
      Promise.resolve(new Response(null, { status: 500 }))) as typeof fetch);

    const { result } = renderHook(() => useChatData("error-chat"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).not.toBeNull();
    });

    expect(result.current.error).toBe("Failed to load chat");
  });

  it("does not fetch when chatId is undefined", () => {
    global.fetch = vi.fn() as typeof fetch;

    renderHook(() => useChatData(undefined), {
      wrapper: createWrapper(),
    });

    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("does not fetch when enabled is false", () => {
    global.fetch = vi.fn() as typeof fetch;

    renderHook(() => useChatData("chat-1", false), {
      wrapper: createWrapper(),
    });

    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("returns empty messages when messages endpoint returns 404", async () => {
    const mockChat = { id: "chat-1", title: "Test" };

    global.fetch = vi.fn(((url: string) => {
      if (url.includes("/messages/")) {
        return Promise.resolve(new Response(null, { status: 404 }));
      }
      if (url.includes("/feedback/")) {
        return Promise.resolve(
          new Response(JSON.stringify({}), { status: 200 }),
        );
      }
      return Promise.resolve(
        new Response(JSON.stringify(mockChat), { status: 200 }),
      );
    }) as typeof fetch);

    const { result } = renderHook(() => useChatData("chat-1"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.chatData).not.toBeUndefined();
    });

    expect(result.current.chatData!.messages).toEqual([]);
    expect(result.current.error).toBeNull();
  });
});
