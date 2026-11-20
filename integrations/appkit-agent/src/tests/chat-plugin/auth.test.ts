import { afterEach, describe, expect, test, vi } from "vitest";

const execFileMock = vi.fn();

vi.mock("node:child_process", () => ({
  execFile: execFileMock,
}));

const originalEnv = { ...process.env };

describe("auth priority", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
    process.env = { ...originalEnv };
  });

  test("prefers CLI (OAuth U2M) over OAuth SP and PAT", async () => {
    process.env.DATABRICKS_HOST = "https://dbc.test";
    process.env.DATABRICKS_CONFIG_PROFILE = "DEFAULT";
    process.env.DATABRICKS_CLIENT_ID = "cid";
    process.env.DATABRICKS_CLIENT_SECRET = "secret";
    process.env.DATABRICKS_TOKEN = "pat";

    const { getAuthMethod } = await import("../../chat-plugin/auth");
    expect(getAuthMethod()).toBe("cli");
  });

  test("prefers OAuth SP over PAT when CLI is not configured", async () => {
    delete process.env.DATABRICKS_CONFIG_PROFILE;
    process.env.DATABRICKS_HOST = "https://dbc.test";
    process.env.DATABRICKS_CLIENT_ID = "cid";
    process.env.DATABRICKS_CLIENT_SECRET = "secret";
    process.env.DATABRICKS_TOKEN = "pat";

    const { getAuthMethod } = await import("../../chat-plugin/auth");
    expect(getAuthMethod()).toBe("oauth");
  });

  test("uses PAT when it is the only configured method", async () => {
    delete process.env.DATABRICKS_CONFIG_PROFILE;
    delete process.env.DATABRICKS_HOST;
    delete process.env.DATABRICKS_CLIENT_ID;
    delete process.env.DATABRICKS_CLIENT_SECRET;
    process.env.DATABRICKS_TOKEN = "pat";

    const { getAuthMethod } = await import("../../chat-plugin/auth");
    expect(getAuthMethod()).toBe("pat");
  });

  test("falls back to PAT when CLI auth is configured but unavailable", async () => {
    process.env.DATABRICKS_HOST = "https://dbc.test";
    process.env.DATABRICKS_TOKEN = "pat";
    delete process.env.DATABRICKS_CLIENT_ID;
    delete process.env.DATABRICKS_CLIENT_SECRET;

    execFileMock.mockImplementation(
      (
        _file: string,
        _args: string[],
        _options: unknown,
        cb: (err: Error | null, stdout: string, stderr: string) => void,
      ) => {
        cb(new Error("databricks CLI unavailable"), "", "databricks not found");
      },
    );

    const { getToken } = await import("../../chat-plugin/auth");
    await expect(getToken()).resolves.toBe("pat");
  });
});
