# Contributing to Agent Bricks CLI and AgentKit

This guide covers the Agent Bricks CLI (`agentbricks`), AgentKit, the runtime, and the project templates,
including how to run and test changes locally and on Databricks Apps. The current Python
distribution is `databricks-agentbricks`; installing it provides the `agentbricks` command and AgentKit.

## Three kinds of change, and how each is sourced

There are three layers a contributor edits. Knowing which one you're changing tells you what to
re-run:

| Layer | What it is | How it's sourced |
| --- | --- | --- |
| **CLI** | the `agentbricks` command (`databricks_agentbricks.cli` and its command modules) | editable install -> runs live from your working tree |
| **Templates** | the project scaffolds under `src/databricks_agentbricks/templates/` | shipped inside the package; `agentbricks init` copies the template matching the installed CLI via `importlib.resources`, which for an editable install resolves to your source tree |
| **SDK / runtime** | `databricks_agentkit.AgentKitClient`, `databricks_agentkit.runtime`, the `langgraph`/`openai` adapters, `DurableAgentServer` | a scaffold depends on the **released** `databricks-agentbricks` distribution from PyPI; opt into local or unreleased code with a `[tool.uv.sources]` override (see below) |

## Editable install (CLI + templates)

```sh
pip install -e integrations/agentbricks     # editable install of the CLI
agentbricks init /tmp/scratch-agent         # scaffolds from your working-tree template
cd /tmp/scratch-agent && agentbricks dev
```

With an editable install, CLI edits and template edits both run straight from your working tree - no
rebuild or commit. Switching branches needs no reinstall, **except** when a branch adds or bumps a
dependency in `integrations/agentbricks/pyproject.toml`:

```sh
pip install -e integrations/agentbricks     # only when dependencies changed
```

Editing a template in the repo only affects **future** `agentbricks init` runs. An existing scaffold has
its own copy of the template, so to iterate on a scaffolded project edit that copy (or re-init).

## Testing SDK / runtime changes in a scaffold

A scaffold uses a normal `databricks-agentbricks` PyPI dependency, so `agentbricks dev` and `agentbricks deploy`
install the **released** SDK - editing the runtime and adapter implementation under `databricks_agentkit` in your
checkout does **not** change what a scaffold runs. To exercise local or unreleased SDK changes, add a
`[tool.uv.sources]` override to the scaffold's `pyproject.toml`. It is a dev-loop-only edit - don't
ship it in a real deployment.

**`agentbricks dev` - your local checkout (editable, picks up uncommitted edits):**

```toml
[tool.uv.sources]
databricks-agentbricks = { path = "/abs/path/to/databricks-ai-bridge/integrations/agentbricks", editable = true }
```

`agentbricks dev` builds the scaffold's venv from this, so your working-tree SDK edits run live. After
changing the pin or the scaffold's dependencies, rebuild once with `agentbricks dev --prepare-environment`
(otherwise `agentbricks dev` reuses the existing `.venv` and you run stale code).

**`agentbricks deploy` - a pushed git ref (the Apps build can't reach a local path):**

```toml
[tool.uv.sources]
databricks-agentbricks = { git = "https://github.com/<you>/databricks-ai-bridge", rev = "<pushed-sha>", subdirectory = "integrations/agentbricks" }
```

Commit and push first - the Apps build clones that commit. A `path` or `file://` pin won't resolve
in the build sandbox, so use a git ref (or a released version) for deploys.

**Verify which SDK a scaffold actually built with** (`direct_url.json` is present when you set an
override):

```sh
# local: the source uv resolved into the agent venv
cat /tmp/scratch-agent/.venv/lib/python*/site-packages/databricks_agentkit-*.dist-info/direct_url.json
# deployed: watch the build/install logs
agentbricks deployments logs agent-bricks-<name>
```

## Keeping docs in sync

[`cli.md`](cli.md) is the CLI command reference - every command, subcommand, argument, and option.
Whenever you change the CLI surface, update `cli.md` in the same change. That includes:

- adding, renaming, or removing a command or subcommand;
- adding, renaming, or removing an argument or option, or changing its default, choices, or whether
  it is required;
- editing a command or option help string, since the reference mirrors it.

Check the tables against the actual `click` definitions in `src/databricks_agentbricks/cli/`; the quickest
drift check is to compare against the CLI's own `--help` output. Purely internal changes that don't
alter the command surface or help text need no `cli.md` update.

## Adding a new agent framework

`agentbricks doctor` recognizes onboarded projects per framework, so adding a new framework (a new
`--framework` choice with its own template and adapter) means updating the doctor's static checks in
the same change, in `src/databricks_agentbricks/cli/doctor.py`:

- add the framework to the `_SUPPORTED_FRAMEWORKS` tuple;
- add its public adapter call symbols to `_FRAMEWORK_ADAPTER_CALLS`, kept in sync with the calls the
  generated template for that framework actually makes (prefix matching is intentionally not used —
  every recognized symbol must be listed explicitly).

Also refresh the framework references and examples in `cli.md` and `README.md`, and add doctor test
coverage for the new framework in `tests/unit_tests/doctor_test.py`.

## Cutting a release

Run **Cut Agent Bricks release** from the Actions tab with a version such as `0.4.0` or
`0.4.1`. Leave **dry_run** on first to see the source commit, branch, tag, and next
development version. Then rerun with dry_run off to make the cut.
The workflow needs permission to write repository contents and open pull requests.

The first `0.4.x` run creates `release/databricks-agentbricks/v0.4` from `main` (or
an explicitly selected ancestor commit). Later runs use the current head of that
branch; merge any required fixes into it before cutting a patch.
The workflow stamps the package version and the minimum Agent Bricks dependency in
all four scaffolds, tests the pushed commit, and tags that commit as
`databricks-agentbricks-v<version>` only after those tests pass. If tests fail, fix
the release branch and rerun with the same version. Once a version is tagged,
use the next patch version for any further fixes (for example, `0.4.1` after `0.4.0`).
Do not move an existing tag.

On the first cut, the workflow also opens a draft PR to change the package version
on `main` to the next minor development version (for example, `0.5.0.dev0` after
cutting `0.4`). That version identifies development builds; it is not a published
release. The scaffolds continue to depend on the released package. A new minor
series starts with a new release branch from `main`. Tagging does not publish the package. Arrange the separate
secure public registry release and its approval to publish to PyPI. The live workspace tool-matrix tests run
separately in the private integration runner and are not included in this gate.
