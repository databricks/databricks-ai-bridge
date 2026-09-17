# Testing local mason changes (`mason dev` / `mason deploy`)

Mason changes fall into **three types**, and each is sourced differently. Knowing which one
you're changing tells you exactly what to re-run. The trap: **the agent SDK is pinned to a
committed git rev, not your working tree** - so SDK edits need commit + re-pin + rebuild.

| Change type | What it is | Where it runs | How it's sourced |
|---|---|---|---|
| **CLI** | `databricks_mason.cli`, `dev.py`, `deploy.py`, `init.py`, `tracing.py` commands | the venv where `mason` is installed | editable install → **live** |
| **SDK / runtime** | `databricks_mason.runtime`, `langgraph`/`openai` adapters, `AgentApp` | the **agent venv** (built by dev/deploy) | scaffold's `[tool.uv.sources] databricks-mason` **frozen git pin** |
| **Template / scaffold** | the scaffold's own `agent/agent.py`, `runtime/main.py`, … | the agent process | copied into the scaffold at `mason init`; afterwards it's just project source |

## Docs: update `cli.md` on any CLI change

[`cli.md`](cli.md) is the command reference for the `mason` CLI - every command, subcommand,
argument, and option, in table form. Users rely on it as the source of truth, so **whenever you
change the CLI surface, update `cli.md` in the same change.** That includes:

- adding, renaming, or removing a command or subcommand;
- adding, renaming, or removing an argument or option, or changing its default, choices, or whether
  it is required;
- editing a command or option help string, since the reference mirrors it.

Keep the tables accurate against the actual `click` definitions in `src/databricks_mason/cli/`. The
quickest drift check is to compare the doc against the CLI's own `--help` output. Purely internal
changes (implementation, refactors) that don't alter the command surface or help text need no
`cli.md` update.

## Setup (once)

Install/point at an **editable** CLI so CLI edits are live (from `integrations/mason`):

```sh
uv sync --all-extras                      # builds .venv with mason installed editable
alias mason="$PWD/.venv/bin/mason"        # or `uv run mason …`
```

## What each change type needs

| Change type | `mason dev` (local build) | `mason deploy` (cloud Apps build) |
|---|---|---|
| **CLI** | Just re-run - editable = live. No rebuild. | Just re-run - editable = live. (Deployed app doesn't contain the CLI.) |
| **Template / scaffold code** | Edit the scaffold's own copy → re-run. **No `--prepare-environment`** (it's app source, not a dep). | Synced every deploy (`sync … --exclude uv.lock`) → deploys as-is. No special step. |
| **SDK / runtime** | Commit → re-pin scaffold → `mason dev --prepare-environment`. `git+file://` works locally. **Fast path:** pin an editable path, rebuild once, then live. | **Commit + push** → pin scaffold to an **`https`** git ref (`file://` is unreachable from the cloud build) → `mason deploy` (fresh build each time). |

## Key rules (why the steps differ)

| Rule | `mason dev` | `mason deploy` |
|---|---|---|
| Working-tree SDK edits picked up? | ❌ pin is a committed rev - must commit | ❌ must commit **and push** |
| Agent venv rebuilt automatically? | ❌ reuses `.venv` unless `--prepare-environment` | ✅ fresh cloud build each deploy |
| `git+file://` SDK pin works? | ✅ local build | ❌ cloud can't reach your machine → use `https` |
| Template-repo edits hit an existing scaffold? | ❌ edit the scaffold's copy, or re-`init` | same |

## Sample commands

CLI change, or template/scaffold-code change - nothing special:

```sh
mason dev --source ./my-agent
mason deploy my-agent --source ./my-agent
```

**SDK / runtime change - `mason dev`:**

```sh
# Fast path: make the agent use your live checkout (edit my-agent/pyproject.toml):
#   [tool.uv.sources]
#   databricks-mason = { path = "<repo-root>/integrations/mason", editable = true }
mason dev --source ./my-agent --prepare-environment   # rebuild once; later .py edits are live

# One-off (frozen commit) alternative:
git commit -am "sdk change"
#   bump `rev` in my-agent/pyproject.toml [tool.uv.sources] databricks-mason to the new SHA, then:
mason dev --source ./my-agent --prepare-environment
```

**SDK / runtime change - `mason deploy`:**

```sh
git commit -am "sdk change" && git push          # cloud build must be able to fetch it
# scaffold fresh against the pushed ref (writes the https pin for you):
mason init ./my-agent --repo https://github.com/databricks/databricks-ai-bridge.git --ref <branch>
#   …or edit my-agent/pyproject.toml [tool.uv.sources] databricks-mason to that git+rev, then:
mason deploy my-agent --source ./my-agent
```

**Verify which SDK commit the agent actually built with:**

```sh
# local: the pin uv resolved into the agent venv
cat ./my-agent/.venv/lib/python*/site-packages/databricks_mason-*.dist-info/direct_url.json
# deployed: watch the build/install
mason deployments logs mason-my-agent
```

## Gotchas that bite

1. **Working-tree SDK edits are invisible** to dev *and* deploy - the pin is a committed git rev. Commit (and, for deploy, push) SDK changes first.
2. **`mason dev` reuses `.venv`** unless `--prepare-environment` - after any dependency/pin change, force the rebuild or you run old SDK code.
3. **`git+file://` is dev-only** - deploy needs an `https` (pushed) ref.
4. **Editing `templates/.../agent.py` in the repo ≠ updating an existing scaffold** - it only affects *future* `mason init`s; to iterate on an existing project, edit its own copy.

> How `mason init` writes the pin: from an editable checkout it pins `databricks-mason` to
> `git+file://<repo>` at your HEAD **at init time** (see `_pin_mason_source` / `_editable_template_source`
> in `src/databricks_mason/init.py`). That's why it's frozen to a commit and never reflects the working tree.
