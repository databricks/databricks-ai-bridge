# Plan: npm Package Management in databricks-ai-bridge

## Current State

The repo is a polyglot monorepo hosting **6 Python packages** and **2 npm packages**:

| Type   | Package                      | Location                        |
|--------|------------------------------|---------------------------------|
| Python | databricks-ai-bridge (core)  | `src/databricks_ai_bridge`      |
| Python | databricks-langchain         | `integrations/langchain`        |
| Python | databricks-openai            | `integrations/openai`           |
| Python | databricks-dspy              | `integrations/dspy`             |
| Python | databricks-llamaindex        | `integrations/llamaindex`       |
| Python | databricks-mcp               | `databricks_mcp`                |
| npm    | @databricks/ai-sdk-provider  | `integrations/ai-sdk-provider`  |
| npm    | @databricks/langchainjs      | `integrations/langchainjs`      |

**Python tooling**: `uv` + `hatchling` + `pyproject.toml`. A `for-each-project` bash script iterates over Python packages. Local dev uses `uv` path sources for editable cross-package installs.

**npm tooling (current)**: Plain `npm` with no workspace or monorepo setup. Each JS package has its own `package-lock.json`. CI workflows delete lockfiles before installing (`rm -f package-lock.json && npm install`). `@databricks/langchainjs` depends on `@databricks/ai-sdk-provider` via a published npm version, not a local link.

### Problems with the Current npm Setup

1. **No shared dependency management** — each package installs independently, duplicating `node_modules`.
2. **Lockfiles are ignored in CI** — `rm -f package-lock.json && npm install` means builds aren't reproducible.
3. **Cross-package development is painful** — to test changes in `ai-sdk-provider` from `langchainjs`, you must publish to npm first (or manually `npm link`).
4. **No unified commands** — no way to build/test/lint all JS packages at once.
5. **Divergent dev dependency versions** — the two packages already have different versions of shared tooling (e.g., `eslint@^8` vs `eslint@^10`, `vitest@^3` vs `vitest@^4`, `tsdown@^0.2` vs `tsdown@^0.9`).

---

## Assessment 1: Package Manager / Monorepo Tooling

### Options Considered

| Tool | Role | Strengths | Weaknesses |
|------|------|-----------|------------|
| **pnpm workspaces** | Package manager + workspace | Fast, disk-efficient, strict deps, native workspace protocol, widely adopted | Requires installing pnpm |
| **npm workspaces** | Package manager + workspace | Ships with Node, zero extra tooling | Slower, phantom dependencies via hoisting, less mature workspace features |
| **yarn (Berry/v4)** | Package manager + workspace | Mature workspaces, Plug'n'Play | PnP breaks many tools, heavier config, less adoption for new projects |
| **Turborepo** | Build orchestrator (on top of pnpm/npm/yarn) | Task caching, parallel execution, dependency-aware ordering | Extra tool, extra config, not a package manager itself |
| **Nx** | Build orchestrator + monorepo framework | Powerful caching, affected-detection, graph visualization | Heavy, opinionated, steep learning curve, overkill here |
| **Lerna** | Monorepo manager | Versioning/publish workflows | Effectively legacy; superseded by native workspaces |

### Recommendation: **pnpm workspaces**

pnpm workspaces is the right choice for this repo. Rationale:

1. **Strict dependency resolution** — pnpm does not hoist packages to the root by default. Each package can only import what it explicitly declares. This is critical for npm libraries: it catches missing `dependencies`/`peerDependencies` entries before users hit them.

2. **Workspace protocol** — `@databricks/langchainjs` can depend on `@databricks/ai-sdk-provider` via `"workspace:*"` during development, automatically resolving to the local copy. At publish time, pnpm rewrites this to the real version. This eliminates the need to publish before testing cross-package changes.

3. **Single lockfile** — one `pnpm-lock.yaml` at the repo root. CI runs `pnpm install --frozen-lockfile` for reproducible builds. Eliminates the current `rm -f package-lock.json` hack.

4. **Disk efficiency** — pnpm's content-addressable store means shared dependencies (TypeScript, vitest, prettier, etc.) are stored once on disk and hard-linked into each package's `node_modules`.

5. **Unified commands** — `pnpm -r run build` builds all packages. `pnpm --filter @databricks/langchainjs run test` targets one. No extra tooling needed.

6. **Ecosystem alignment** — the Vercel AI SDK (which `ai-sdk-provider` integrates with) itself uses pnpm. LangChain.js uses yarn, but pnpm is interoperable.

7. **Incremental adoption** — pnpm workspaces require minimal config (`pnpm-workspace.yaml` + root `package.json`). Turborepo can be layered on top later if CI caching becomes valuable as the number of packages grows.

### Why Not Turborepo (Yet)?

Turborepo adds value when you have many packages with expensive build/test steps and want CI cache hits across runs. With 2-3 npm packages, `pnpm --filter` and GitHub Actions path-based triggering (already in place) provide sufficient scoping. The overhead of configuring `turbo.json` pipeline definitions, remote caching, and the extra dependency isn't justified today. It's a natural evolution step if the JS side grows to 5+ packages.

### Why Not npm Workspaces?

npm workspaces would avoid adding a new tool, but the tradeoffs are unfavorable:
- Phantom dependencies via hoisting (dangerous for published libraries)
- Slower install/resolution times
- The `workspace:` protocol is supported but less mature
- No `--frozen-lockfile` equivalent as robust as pnpm's

---

## Assessment 2: Accommodating npm Packages in a Python-First Repo

### Principles

The Python and JS ecosystems should be **independent but cohabiting**: separate toolchains, separate CI pipelines, separate lockfiles, but sharing a directory structure and git history. Neither side should require the other's tooling to work.

### Proposed Directory Structure

No reorganization of existing directories is needed. The current layout already works:

```
databricks-ai-bridge/
├── package.json                  # NEW — pnpm workspace root (private: true)
├── pnpm-workspace.yaml           # NEW — declares JS workspace packages
├── pnpm-lock.yaml                # NEW — single lockfile for all JS packages
├── pyproject.toml                # Existing — Python core package
├── for-each-project              # Existing — Python-only iteration script
│
├── src/                          # Python core source
├── databricks_mcp/               # Python MCP package
├── integrations/
│   ├── langchain/                # Python
│   ├── openai/                   # Python
│   ├── dspy/                     # Python
│   ├── llamaindex/               # Python
│   ├── ai-sdk-provider/          # npm — @databricks/ai-sdk-provider
│   └── langchainjs/              # npm — @databricks/langchainjs
├── tests/                        # Python core tests
└── docs/                         # Sphinx docs
```

Key new files:

**`pnpm-workspace.yaml`**:
```yaml
packages:
  - "integrations/ai-sdk-provider"
  - "integrations/langchainjs"
```

**Root `package.json`** (workspace root — never published):
```json
{
  "private": true,
  "engines": { "node": ">=18.0.0" },
  "packageManager": "pnpm@10.x",
  "scripts": {
    "build": "pnpm -r run build",
    "test": "pnpm -r run test",
    "lint": "pnpm -r run lint",
    "typecheck": "pnpm -r run typecheck",
    "format:check": "pnpm -r run format:check"
  }
}
```

### Isolation Between Python and JS

| Concern | Python | JS |
|---------|--------|----|
| Package manager | `uv` | `pnpm` |
| Lockfile | `uv.lock` (per-project, gitignored) | `pnpm-lock.yaml` (root, committed) |
| Build system | hatchling | tsdown |
| Linting | ruff | eslint |
| Type checking | ty | tsc |
| Testing | pytest | vitest |
| Iteration script | `for-each-project` | `pnpm -r` / `pnpm --filter` |
| CI trigger | Path-filtered workflows | Path-filtered workflows (already exists) |

There is no coupling: Python contributors never need Node installed, and JS contributors never need Python/uv.

### Changes Required

#### 1. Add root workspace config (low effort)

Add `pnpm-workspace.yaml` and root `package.json` as shown above.

#### 2. Update `.gitignore`

Add explicit `node_modules/` entry (currently not present — the existing `.gitignore` doesn't list it, though packages have their own). Add `pnpm-lock.yaml` removal from gitignore if needed (it should be committed).

```gitignore
# Node
node_modules/
```

#### 3. Update inter-package dependency to use workspace protocol

In `integrations/langchainjs/package.json`, change:
```json
"@databricks/ai-sdk-provider": "^0.5.1"
```
to:
```json
"@databricks/ai-sdk-provider": "workspace:^"
```

pnpm will resolve this to the local package during development and rewrite it to the real version range at publish time.

#### 4. Delete per-package lockfiles

Remove `integrations/langchainjs/package-lock.json` and `integrations/ai-sdk-provider/package-lock.json`. The single root `pnpm-lock.yaml` replaces them.

#### 5. Update CI workflows

Replace the npm-based install steps:

```yaml
# Before
- name: Install dependencies
  run: rm -f package-lock.json && npm install

# After
- name: Install pnpm
  uses: pnpm/action-setup@v4

- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: ${{ matrix.node-version }}
    cache: "pnpm"

- name: Install dependencies
  run: pnpm install --frozen-lockfile
```

The `--frozen-lockfile` flag ensures CI fails if the lockfile is out of date, enforcing reproducibility.

Since pnpm installs the whole workspace in one `pnpm install`, but builds/tests can still be scoped with `--filter`, the workflow `working-directory` defaults can remain. Package scripts (`npm run test` → `pnpm run test`) work identically.

#### 6. Harmonize shared dev tooling (optional, recommended)

The two JS packages have divergent versions of shared tools. With pnpm workspaces, you can optionally centralize these via the workspace root's `devDependencies` or a shared config package. Immediate candidates:

- **TypeScript**: standardize on one version
- **Prettier**: share config via root `.prettierrc`
- **ESLint**: share config via root `eslint.config.js` (flat config)
- **vitest**: standardize version

This is not blocking but reduces drift and simplifies upgrades.

#### 7. Adding future JS packages

To add a new npm package:
1. Create the directory (e.g., `integrations/new-package/`)
2. Add `package.json` with the standard scripts
3. Add the path to `pnpm-workspace.yaml`
4. Run `pnpm install` from the repo root
5. Add a CI workflow (copy an existing one, update paths)

No changes to the Python side needed.

### What NOT to Change

- **`for-each-project`** — keep as-is for Python. No need to make it JS-aware.
- **Python `pyproject.toml` files** — completely independent, no changes.
- **Python CI workflows** — completely independent, no changes.
- **Repo-level `pyproject.toml`** — the root `package.json` coexists without conflict.

---

## Summary of Recommendations

| Decision | Recommendation |
|----------|---------------|
| JS package manager | **pnpm** (with workspaces) |
| Monorepo build orchestrator | **None for now** (pnpm --filter is sufficient; add Turborepo later if needed) |
| JS package location | **`integrations/`** directory (consistent with current layout) |
| Cross-package deps | **`workspace:^`** protocol |
| CI reproducibility | **`pnpm install --frozen-lockfile`** (replaces `rm -f package-lock.json && npm install`) |
| Python/JS coupling | **None** — fully independent toolchains sharing a git repo |

### Estimated Effort

The migration from standalone npm to pnpm workspaces is low-risk and can be done in a single PR:
- Add 2 new files (`pnpm-workspace.yaml`, root `package.json`)
- Delete 2 lockfiles
- Edit 2 CI workflows
- Edit 1 line in `langchainjs/package.json`
- Add 1 line to `.gitignore`
