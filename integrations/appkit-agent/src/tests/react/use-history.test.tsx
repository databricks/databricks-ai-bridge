// @vitest-environment happy-dom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import {
  getChatHistoryPaginationKey,
  useHistory,
} from "../../react/hooks/use-history";
import { createWrapper } from "./test-helpers";

describe("getChatHistoryPaginationKey", () => {
  it("returns first page URL when pageIndex is 0", () => {
    const key = getChatHistoryPaginationKey("/api/chat", 0, undefined);
    expect(key).toBe("/api/chat/history?limit=20");
  });

  it("returns null when previous page has hasMore=false", () => {
    const key = getChatHistoryPaginationKey("/api/chat", 1, {
      chats: [],
      hasMore: false,
    });
    expect(key).toBeNull();
  });

  it("returns cursor URL based on last chat id", () => {
    const prev = {
      chats: [{ id: "chat-1" }, { id: "chat-2" }],
      hasMore: true,
    };
    const key = getChatHistoryPaginationKey("/api/chat", 1, prev);
    expect(key).toBe("/api/chat/history?ending_before=chat-2&limit=20");
  });

  it("returns null when previous page has no chats", () => {
    const key = getChatHistoryPaginationKey("/api/chat", 1, {
      chats: [],
      hasMore: true,
    });
    expect(key).toBeNull();
  });

  it("respects custom apiBase", () => {
    const key = getChatHistoryPaginationKey("/custom/api", 0, undefined);
    expect(key).toBe("/custom/api/history?limit=20");
  });
});

describe("useHistory", () => {
  const originalFetch = global.fetch;

  function mockFetch(
    overrides: Record<string, (url: string, init?: RequestInit) => Promise<Response>> = {},
  ) {
    global.fetch = vi.fn(((url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (overrides[method]) return overrides[method](url, init);

      // Default: return empty history page for SWR fetches
      return Promise.resolve(
        new Response(JSON.stringify({ chats: [], hasMore: false }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }) as typeof fetch);
  }

  beforeEach(() => {
    mockFetch();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("deleteChat throws on non-2xx response", async () => {
    mockFetch({
      DELETE: () =>
        Promise.resolve(new Response(null, { status: 500 })),
    });

    const { result } = renderHook(() => useHistory(), {
      wrapper: createWrapper(),
    });

    await expect(result.current.deleteChat("test-id")).rejects.toThrow(
      "Failed to delete chat: 500",
    );
  });

  it("deleteChat succeeds on 2xx and calls correct URL", async () => {
    mockFetch({
      DELETE: () =>
        Promise.resolve(new Response(null, { status: 200 })),
    });

    const { result } = renderHook(() => useHistory(), {
      wrapper: createWrapper(),
    });

    await expect(
      result.current.deleteChat("chat-123"),
    ).resolves.not.toThrow();

    expect(global.fetch).toHaveBeenCalledWith("/api/chat/chat-123", {
      method: "DELETE",
    });
  });

  it("renameChat throws on non-2xx response", async () => {
    mockFetch({
      PATCH: () =>
        Promise.resolve(new Response(null, { status: 403 })),
    });

    const { result } = renderHook(() => useHistory(), {
      wrapper: createWrapper(),
    });

    await expect(
      result.current.renameChat("test-id", "New Title"),
    ).rejects.toThrow("Failed to rename chat: 403");
  });

  it("renameChat sends PATCH with title in body", async () => {
    mockFetch({
      PATCH: () =>
        Promise.resolve(new Response(null, { status: 200 })),
    });

    const { result } = renderHook(() => useHistory(), {
      wrapper: createWrapper(),
    });

    await result.current.renameChat("chat-456", "New Title");

    expect(global.fetch).toHaveBeenCalledWith("/api/chat/chat-456", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New Title" }),
    });
  });

  it("returns empty state when chat history is disabled", () => {
    const { result } = renderHook(() => useHistory(), {
      wrapper: createWrapper({ chatHistoryEnabled: false }),
    });

    expect(result.current.chats).toEqual([]);
    expect(result.current.isEmpty).toBe(true);
  });
});
