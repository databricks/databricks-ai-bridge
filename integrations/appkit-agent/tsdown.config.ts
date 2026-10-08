import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    index: "./src/index.ts",
  },
  format: ["esm", "cjs"],
  fixedExtension: true,
  dts: true,
  outDir: "./dist",
  target: "esnext",
  clean: true,
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
});
