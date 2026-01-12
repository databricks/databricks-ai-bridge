import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/**/*.int.test.ts"],
    exclude: ["node_modules", "dist"],
    testTimeout: 120000,
    hookTimeout: 60000,
  },
});
