import { defineConfig } from 'tsdown'

export default defineConfig({
  // Entry points for each export path
  entry: {
    index: './src/index.ts',
  },

  // Output formats
  format: ['esm', 'cjs'],

  // Required so tsdown emits .mjs/.cjs files matching the explicit extensions in package.json "exports"
  fixedExtension: true,

  // Generate TypeScript declarations
  dts: true,

  // Output directory
  outDir: './dist',

  // Target modern JavaScript
  target: 'esnext',

  // Clean output directory before build
  clean: true,

  // Generate sourcemaps for debugging
  sourcemap: true,

  // External dependencies (should not be bundled)
  external: ['@ai-sdk/provider', '@ai-sdk/provider-utils', 'zod'],

  // Platform target
  platform: 'node',

  // Tree-shake unused dependencies
  treeshake: true,
})
