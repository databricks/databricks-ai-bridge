import { describe, expect, test } from "vitest";
import { generateUUID, truncatePreserveWords } from "../../chat-plugin/utils";

describe("generateUUID", () => {
  test("returns a v4 UUID-shaped string", () => {
    const uuid = generateUUID();
    expect(uuid).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    );
  });

  test("returns unique values", () => {
    const ids = new Set(Array.from({ length: 100 }, () => generateUUID()));
    expect(ids.size).toBe(100);
  });
});

describe("truncatePreserveWords", () => {
  test("returns input unchanged if shorter than maxLength", () => {
    expect(truncatePreserveWords("hello world", 100)).toBe("hello world");
  });

  test("truncates at word boundary", () => {
    expect(truncatePreserveWords("hello wonderful world", 14)).toBe("hello");
    expect(truncatePreserveWords("hello wonderful world", 16)).toBe(
      "hello wonderful",
    );
  });

  test("returns empty for maxLength 0", () => {
    expect(truncatePreserveWords("hello", 0)).toBe("");
  });
});
