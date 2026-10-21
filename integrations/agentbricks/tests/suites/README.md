# Agent Bricks E2E suites

Black-box tests of the `agentbricks` CLI, at the top of the test pyramid: one happy-path scenario
per command group, each running the real CLI and checking the agent it produces. Error cases and
edge cases belong in the unit tests.

## Running

From `integrations/agentbricks`:

```bash
uv run pytest tests/suites -n 6 --dist loadgroup \
    --databricks-profile <profile> --catalog <catalog>
```

- Narrow it to a file (`tests/suites/test_deploy.py`) or a test.
- `-n 6 --dist loadgroup` runs tests in parallel; `loadgroup` keeps the dev tests on one worker,
  because local dev servers share a proxy port.
- Workspace resources the suite borrows but never creates are flags: `--catalog`,
  `--warehouse-id`, and `--genie-space-id` (without one the Genie tool is skipped).
- To set flags once for a shell or a CI job, use pytest's own `PYTEST_ADDOPTS`, e.g.
  `export PYTEST_ADDOPTS="--databricks-profile <profile> --catalog <catalog>"`.
- `--model` sets the model every invocation asks for (workspaces name models differently, e.g.
  `system.ai.databricks-gpt-5-2` on staging).
- `--workspace fake` runs against no workspace; everything that needs one is skipped.
- `--keep-resources` keeps leftovers for debugging; each test still deletes its App and stores.

The run id and output directory (logs, generated projects, `apps.jsonl`) are printed at the start:
`Agent Bricks E2E output: /tmp/agentbricks-e2e-<id> (run <id>)`.

## Cleanup

Every resource a run creates carries its run id, and is deleted at three points:

1. **When each test ends**, the test's own cleanup runs. Each test registers the delete command
   for everything it creates with the `cleanup` fixture, like Go's `t.Cleanup`, e.g.
   `cleanup(f"deployments delete {app} --delete-source --yes")`; registered commands run newest
   first, pass or fail.
   Fixtures delete what they create, such as `scratch_schema` and its schema.
2. **When the run ends**, `pytest_sessionfinish` deletes everything carrying the run's name prefix
   that is still there (stores, Apps, synced source folders, tracing experiments, schemas) and the
   Apps' Lakebase roles (`harness.cleanup.delete_run_leftovers`). `deployments delete` leaves the
   experiments and roles behind; the sweep catches a source folder only when the App wasn't
   confirmed gone or its delete never ran.
   A failure there fails the run.
3. **After a run that was killed** before step 2, for example by a CI timeout:

   ```bash
   uv run pytest tests/suites --cleanup-only --run-id <id> --databricks-profile <profile> --catalog <catalog>
   ```

   CI passes its own `--run-id` and runs this in an `always()` step.

## Layout

| Path | What it is |
|---|---|
| `tests/suites/conftest.py` | All pytest wiring: options, hooks, and the fixtures every suite shares |
| `tests/harness/run.py` | `RunConfig`, the run id, and the name prefixes every resource carries |
| `tests/harness/target.py` | The `--workspace live\|fake` seam the CLI and fixtures talk to |
| `tests/harness/agentbricks_cli.py` | Runs `agentbricks` commands: `cli.run("tools add ...")`, and `cli.start("dev ...")` in the background; `cli.invoke(app, prompt)` runs `endpoint invoke` with the `--model` the run asks for |
| `tests/harness/project.py` | A test's project: run-scoped path and App name, its files, and `pin_runtime()` |
| `tests/harness/workspace.py` | Databricks SDK helpers for what is not under test: prerequisites (UC objects, warehouse, Genie) and cleanup no CLI command does yet |
| `tests/harness/cleanup.py` | Deleting what the suites create: per test through the CLI, and per run for the rest |
| `tests/harness/logs.py` | Test-log output; every command is echoed so a run's log replays it |
| `tests/suites/test_<group>.py` | One file per CLI command group: `test_tools`, `test_dev`, `test_deploy`, `test_memory`, `test_sessions`, `test_tracing` |
| `tests/suites/tool_definitions.py`, `uc_objects.py` | The tools `test_tools`, `test_dev` and `test_deploy` add, and the UC objects they use |

`conftest.py` files are the only code pytest calls by name. `tests/harness/` sits next to `tests/suites/`
rather than inside it: it is plain Python the suites import as `harness` (via `pythonpath` in
`pyproject.toml`), not a suite.

## Adding things

- **A tool:** add one `ToolDefinition(name, requires, build)` in `tool_definitions.py`.
  `requires` names the fixtures it needs; tools whose fixtures skip are left out and logged.
- **A command group:** add `test_<group>.py` with its one happy path. A test takes the `project` fixture, runs
  `agentbricks_cli.run(f"init ... {project.path}")`, then `project.pin_runtime()` so the project
  runs the checkout's runtime; teardown fails a test that skips the pin. Register a `cleanup(...)`
  delete command for everything the test creates, before the step that creates it if a failed step
  can leave it behind. Resources the CLI under test does not create itself are fixtures in
  `conftest.py`, created per test and deleted in the fixture's teardown.
