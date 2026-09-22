# Contributing to `databricks-mason`

This guide covers developing Mason itself - the CLI, the SDK/runtime, and the project templates -
and how to run and test your changes locally and on Databricks Apps.

## Three kinds of change, and how each is sourced

Mason has three layers a contributor edits. Knowing which one you're changing tells you what to
re-run:

| Layer | What it is | How it's sourced |
| --- | --- | --- |
| **CLI** | the `mason` command (`databricks_mason.cli` and its command modules) | editable install -> runs live from your working tree |
| **Templates** | the project scaffolds under `src/databricks_mason/templates/` | shipped inside the package; `mason init` copies the template matching the installed CLI via `importlib.resources`, which for an editable install resolves to your source tree |
| **SDK / runtime** | `databricks_mason.runtime`, the `langgraph`/`openai` adapters, `AgentApp` | a scaffold depends on the **released** `databricks-mason` from PyPI; opt into local or unreleased code with a `[tool.uv.sources]` override (see below) |

## Editable install (CLI + templates)

```sh
pip install -e integrations/mason     # editable install of the CLI
mason init /tmp/scratch-agent         # scaffolds from your working-tree template
cd /tmp/scratch-agent && mason dev
```

With an editable install, CLI edits and template edits both run straight from your working tree - no
rebuild or commit. Switching branches needs no reinstall, **except** when a branch adds or bumps a
dependency in `integrations/mason/pyproject.toml`:

```sh
pip install -e integrations/mason     # only when dependencies changed
```

Editing a template in the repo only affects **future** `mason init` runs. An existing scaffold has
its own copy of the template, so to iterate on a scaffolded project edit that copy (or re-init).

## Testing SDK / runtime changes in a scaffold

A scaffold uses a normal `databricks-mason` PyPI dependency, so `mason dev` and `mason deploy`
install the **released** SDK - editing `databricks_mason.runtime` / `.langgraph` / `.openai` in your
checkout does **not** change what a scaffold runs. To exercise local or unreleased SDK changes, add a
`[tool.uv.sources]` override to the scaffold's `pyproject.toml`. It is a dev-loop-only edit - don't
ship it in a real deployment.

**`mason dev` - your local checkout (editable, picks up uncommitted edits):**

```toml
[tool.uv.sources]
databricks-mason = { path = "/abs/path/to/databricks-ai-bridge/integrations/mason", editable = true }
```

`mason dev` builds the scaffold's venv from this, so your working-tree SDK edits run live. After
changing the pin or the scaffold's dependencies, rebuild once with `mason dev --prepare-environment`
(otherwise `mason dev` reuses the existing `.venv` and you run stale code).

**`mason deploy` - a pushed git ref (the Apps build can't reach a local path):**

```toml
[tool.uv.sources]
databricks-mason = { git = "https://github.com/<you>/databricks-ai-bridge", rev = "<pushed-sha>", subdirectory = "integrations/mason" }
```

Commit and push first - the Apps build clones that commit. A `path` or `file://` pin won't resolve
in the build sandbox, so use a git ref (or a released version) for deploys.

**Verify which SDK a scaffold actually built with** (`direct_url.json` is present when you set an
override):

```sh
# local: the source uv resolved into the agent venv
cat /tmp/scratch-agent/.venv/lib/python*/site-packages/databricks_mason-*.dist-info/direct_url.json
# deployed: watch the build/install logs
mason deployments logs agent-mason-<name>
```

## Keeping docs in sync

[`cli.md`](cli.md) is the CLI command reference - every command, subcommand, argument, and option.
Whenever you change the CLI surface, update `cli.md` in the same change. That includes:

- adding, renaming, or removing a command or subcommand;
- adding, renaming, or removing an argument or option, or changing its default, choices, or whether
  it is required;
- editing a command or option help string, since the reference mirrors it.

Check the tables against the actual `click` definitions in `src/databricks_mason/cli/`; the quickest
drift check is to compare against the CLI's own `--help` output. Purely internal changes that don't
alter the command surface or help text need no `cli.md` update.

## Testing

- Mocks give false confidence about library behavior; verify real API semantics with a live/E2E run.
- Reproduce a bug end-to-end before fixing, re-verify the fix live, and clean up all created resources
  + kill local dev servers afterward.

## PR etiquette

- Append-only; reconcile stacked branches by merge, not rebase.
- Update docs (`cli.md` / README) in the same change; keep `cli.md` byte-matched to `--help` (see
  [Keeping docs in sync](#keeping-docs-in-sync)).
- Run `/review` before/after push and re-run after material changes (a prior approve goes stale).
- Reply to comments at the boundary they concern; use a `TODO`, not a silent workaround, when deferring.
