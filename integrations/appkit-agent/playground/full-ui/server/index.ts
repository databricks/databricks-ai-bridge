import { createApp, server } from "@databricks/appkit";
import { agent, chat } from "@databricks/appkit-agent";

await createApp({
  plugins: [
    server({ autoStart: true, staticPath: chat.staticAssetsPath }),
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
