import { defineConfig } from "tsdown";

export default defineConfig([
  {
    entry: {
      index: "./src/index.ts",
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
      "@databricks/langchainjs",
      "@langchain/core",
      "@langchain/langgraph",
      "@langchain/mcp-adapters",
      "express",
    ],
    platform: "node",
    treeshake: true,
  },
  {
    entry: {
      "chat-ui/simple": "./src/chat-ui/simple/index.ts",
    },
    format: ["esm", "cjs"],
    fixedExtension: true,
    dts: true,
    outDir: "./dist",
    target: "esnext",
    clean: false,
    sourcemap: true,
    external: [
      "react",
      "react-dom",
      "react/jsx-runtime",
      "@radix-ui/react-slot",
      "class-variance-authority",
      "clsx",
      "tailwind-merge",
    ],
    platform: "browser",
    treeshake: true,
  },
]);
