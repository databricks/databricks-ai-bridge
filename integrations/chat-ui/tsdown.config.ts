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
  external: ["react", "react-dom", "react/jsx-runtime"],
  platform: "browser",
  treeshake: true,
});
