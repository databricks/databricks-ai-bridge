import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  base: "/",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@databricks/appkit-agent/react": path.resolve(
        __dirname,
        "../src/react",
      ),
    },
    dedupe: ["react", "react-dom", "swr", "@ai-sdk/react", "ai"],
  },
  build: {
    outDir: "../dist/chat-client",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      external: [],
    },
  },
  optimizeDeps: {
    include: ["swr", "swr/infinite", "@ai-sdk/react", "ai"],
  },
});
