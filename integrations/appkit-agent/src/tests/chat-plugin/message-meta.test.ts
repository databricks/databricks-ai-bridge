import { describe, expect, test } from "vitest";
import {
  getMessageMetadata,
  getMessageMetasByChatId,
  storeMessageMeta,
} from "../../chat-plugin/message-meta";

describe("message-meta", () => {
  test("stores and retrieves metadata", () => {
    storeMessageMeta("msg-1", "chat-1", "trace-abc");
    const meta = getMessageMetadata("msg-1");
    expect(meta).toEqual({ traceId: "trace-abc", chatId: "chat-1" });
  });

  test("returns null for unknown message", () => {
    expect(getMessageMetadata("nonexistent")).toBeNull();
  });

  test("getMessageMetasByChatId filters by chatId", () => {
    storeMessageMeta("msg-a", "chat-x", "trace-1");
    storeMessageMeta("msg-b", "chat-x", "trace-2");
    storeMessageMeta("msg-c", "chat-y", "trace-3");

    const metas = getMessageMetasByChatId("chat-x");
    expect(metas.length).toBeGreaterThanOrEqual(2);
    expect(metas.every((m) => m.traceId !== undefined)).toBe(true);
  });

  test("skips entries with null traceId", () => {
    storeMessageMeta("msg-null", "chat-null", null);
    const metas = getMessageMetasByChatId("chat-null");
    expect(metas.find((m) => m.messageId === "msg-null")).toBeUndefined();
  });
});
