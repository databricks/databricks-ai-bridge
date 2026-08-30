import { defineConfig } from "tsdown";

export default [
  defineConfig({
    entry: {
      index: "./src/index.ts",
      "chat/index": "./src/chat-plugin/index.ts",
    },
    format: ["esm", "cjs"],
    fixedExtension: true,
    dts: true,
    outDir: "./dist",
    target: "esnext",
    clean: false,
    sourcemap: true,
    external: [
      "@databricks/appkit",
      "@databricks/ai-sdk-provider",
      "@databricks/langchainjs",
      "@langchain/core",
      "@langchain/langgraph",
      "@langchain/mcp-adapters",
      "express",
      "pg",
    ],
    platform: "node",
    treeshake: true,
  }),
  defineConfig({
    entry: { "react/index": "./src/react/index.ts" },
    format: ["esm", "cjs"],
    fixedExtension: true,
    dts: true,
    outDir: "./dist",
    target: "esnext",
    clean: false,
    sourcemap: true,
    platform: "neutral",
    external: [
      "react",
      "react-dom",
      "swr",
      "swr/infinite",
      "@ai-sdk/react",
      "ai",
    ],
    treeshake: true,
  }),
];
