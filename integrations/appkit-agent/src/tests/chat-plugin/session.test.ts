import { afterEach, describe, expect, test, vi } from "vitest";
import { createMockRequest } from "./test-helpers";

const getScimUserMock = vi.fn();

vi.mock("../../chat-plugin/auth", () => ({
  getScimUser: getScimUserMock,
}));

const originalEnv = { ...process.env };

describe("resolveSession", () => {
  afterEach(() => {
    vi.clearAllMocks();
    process.env = { ...originalEnv };
  });

  test("prefers custom getSession callback", async () => {
    const { resolveSession } = await import("../../chat-plugin/session");
    const req = createMockRequest({
      headers: {
        "x-forwarded-user": "forwarded-user",
      },
    });

    const session = await resolveSession(req as never, () => ({
      user: { id: "custom-user", email: "custom@databricks.com" },
    }));

    expect(session).toEqual({
      user: { id: "custom-user", email: "custom@databricks.com" },
    });
    expect(getScimUserMock).not.toHaveBeenCalled();
  });

  test("uses forwarded user headers when custom callback is not provided", async () => {
    const { resolveSession } = await import("../../chat-plugin/session");
    const req = createMockRequest({
      headers: {
        "x-forwarded-user": "forwarded-user",
        "x-forwarded-email": "forwarded@databricks.com",
        "x-forwarded-preferred-username": "forwarded-name",
      },
    });

    const session = await resolveSession(req as never);

    expect(session).toEqual({
      user: {
        id: "forwarded-user",
        email: "forwarded@databricks.com",
        name: "forwarded-name",
        preferredUsername: "forwarded-name",
      },
    });
    expect(getScimUserMock).not.toHaveBeenCalled();
  });

  test("falls back to SCIM user when forwarded headers are absent", async () => {
    const { resolveSession } = await import("../../chat-plugin/session");
    getScimUserMock.mockResolvedValueOnce({
      id: "scim-user",
      email: "scim@databricks.com",
      name: "Scim User",
      preferredUsername: "scim-user",
    });

    const session = await resolveSession(createMockRequest() as never);

    expect(session).toEqual({
      user: {
        id: "scim-user",
        email: "scim@databricks.com",
        name: "Scim User",
        preferredUsername: "scim-user",
      },
    });
  });

  test("falls back to local USER when SCIM is unavailable", async () => {
    const { resolveSession } = await import("../../chat-plugin/session");
    getScimUserMock.mockResolvedValueOnce(null);
    process.env.USER = "local-user";

    const session = await resolveSession(createMockRequest() as never);

    expect(session).toEqual({
      user: {
        id: "local-user",
        email: "local-user@localhost",
        name: "local-user",
      },
    });
  });
});
