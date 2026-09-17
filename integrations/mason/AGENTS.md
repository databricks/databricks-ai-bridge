# Mason agent guide

Instructions for agents working in `integrations/mason`.

## Developing and testing Mason

For the local dev loop - the editable install, how CLI / SDK-runtime / template changes are each
sourced, testing unreleased SDK changes with a `[tool.uv.sources]` override, and the `mason dev` /
`mason deploy` gotchas - follow [CONTRIBUTING.md](CONTRIBUTING.md). Key reminder: a scaffold installs
the **released** `databricks-mason`, so working-tree SDK edits only take effect through a
`[tool.uv.sources]` override (an editable `path` for `mason dev`; a pushed `git` ref for
`mason deploy`).

## Keep `cli.md` current

[`cli.md`](cli.md) is the CLI command reference. Whenever you change the CLI surface - add, rename,
or remove a command, subcommand, argument, or option; change a default, choices, or whether an option
is required; or edit a command/option help string - update `cli.md` in the same change, checked
against `mason --help` and the `click` definitions in `src/databricks_mason/cli/`. See
[CONTRIBUTING.md](CONTRIBUTING.md#keeping-docs-in-sync) for the full rule.
