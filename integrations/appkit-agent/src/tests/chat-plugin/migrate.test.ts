import { describe, expect, test, vi } from "vitest";

vi.mock("drizzle-orm/node-postgres", () => ({
  drizzle: vi.fn(() => "__mock_db__"),
}));

vi.mock("drizzle-orm/node-postgres/migrator", () => ({
  migrate: vi.fn().mockResolvedValue(undefined),
}));

import { ensureSchema } from "../../chat-plugin/migrate";
import { drizzle } from "drizzle-orm/node-postgres";
import { migrate } from "drizzle-orm/node-postgres/migrator";

describe("ensureSchema", () => {
  test("calls drizzle migrate with the pool and migrations folder", async () => {
    const pool = { query: vi.fn() };
    await ensureSchema(pool as never);

    expect(drizzle).toHaveBeenCalledWith(pool);
    expect(migrate).toHaveBeenCalledTimes(1);
    expect(migrate).toHaveBeenCalledWith("__mock_db__", {
      migrationsFolder: expect.stringContaining("drizzle"),
    });
  });

  test("propagates migration errors", async () => {
    vi.mocked(migrate).mockRejectedValueOnce(new Error("connection refused"));
    const pool = { query: vi.fn() };
    await expect(ensureSchema(pool as never)).rejects.toThrow(
      "connection refused",
    );
  });
});
