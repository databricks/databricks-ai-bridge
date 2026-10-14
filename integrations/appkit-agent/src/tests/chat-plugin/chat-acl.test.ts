import { describe, expect, test } from "vitest";
import { checkChatAccess } from "../../chat-plugin/chat-acl";

const makeChat = (
  overrides: Partial<{
    id: string;
    userId: string;
    visibility: "public" | "private";
    title: string;
    createdAt: Date;
    lastContext: null;
  }> = {},
) => ({
  id: "chat-1",
  userId: "user-1",
  visibility: "private" as const,
  title: "Test",
  createdAt: new Date(),
  lastContext: null,
  ...overrides,
});

describe("checkChatAccess", () => {
  test("returns not_found when chat does not exist", async () => {
    const result = await checkChatAccess(async () => null, "chat-1", "user-1");
    expect(result.allowed).toBe(false);
    expect(result.reason).toBe("not_found");
  });

  test("allows owner of private chat", async () => {
    const chat = makeChat({ userId: "user-1", visibility: "private" });
    const result = await checkChatAccess(async () => chat, "chat-1", "user-1");
    expect(result.allowed).toBe(true);
  });

  test("denies non-owner of private chat", async () => {
    const chat = makeChat({ userId: "user-1", visibility: "private" });
    const result = await checkChatAccess(async () => chat, "chat-1", "user-2");
    expect(result.allowed).toBe(false);
    expect(result.reason).toBe("forbidden");
  });

  test("allows anyone for public chat", async () => {
    const chat = makeChat({ userId: "user-1", visibility: "public" });
    const result = await checkChatAccess(async () => chat, "chat-1", "user-2");
    expect(result.allowed).toBe(true);
  });
});
