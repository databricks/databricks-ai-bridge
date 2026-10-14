import { describe, expect, test } from "vitest";
import { ChatServerError } from "../../chat-plugin/errors";

describe("ChatServerError", () => {
  test("returns correct status for known codes", () => {
    const cases: Array<[string, number]> = [
      ["bad_request:api", 400],
      ["unauthorized:chat", 401],
      ["forbidden:chat", 403],
      ["not_found:database", 404],
      ["empty:stream", 204],
      ["offline:chat", 503],
      ["bad_request:database", 500],
    ];
    for (const [code, expectedStatus] of cases) {
      const err = new ChatServerError(code);
      const { status } = err.toResponse();
      expect(status).toBe(expectedStatus);
    }
  });

  test("returns 500 for unknown codes", () => {
    const err = new ChatServerError("unknown:code");
    const { status } = err.toResponse();
    expect(status).toBe(500);
  });

  test("uses custom message in response", () => {
    const err = new ChatServerError("forbidden:chat", "Not your chat");
    const { json } = err.toResponse();
    expect(json.error).toBe("Not your chat");
    expect(json.code).toBe("forbidden:chat");
  });
});
