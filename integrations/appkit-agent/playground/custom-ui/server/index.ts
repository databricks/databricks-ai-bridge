import fs from "node:fs";
import path from "node:path";
import { createApp, server } from "@databricks/appkit";
import { agent, chat } from "@databricks/appkit-agent";

const clientBuildPath = path.resolve(import.meta.dirname, "..", "dist/client");
const staticPath = fs.existsSync(clientBuildPath)
  ? clientBuildPath
  : undefined;

await createApp({
  plugins: [
    server({ autoStart: true, ...(staticPath && { staticPath }) }),
    agent({
      model: process.env.DATABRICKS_MODEL || "databricks-claude-sonnet-4-5",
      systemPrompt: "You are a helpful assistant.",
      tools: [],
    }),
    chat({
      backend: "agent",
    }),
  ],
});
