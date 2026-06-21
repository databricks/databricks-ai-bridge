import { defineConfig } from "drizzle-kit";

export default defineConfig({
  schema: "./src/chat-plugin/schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
});
