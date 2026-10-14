# Agent Bricks E2E suites

Black-box tests of the `agentbricks` CLI: each test runs the real CLI (`init`, `tools add`, `dev`,
`deploy`, `endpoint invoke`) and checks the agent it produces.

## Running

From `integrations/agentbricks`:

```bash
uv run pytest tests/suites -n 6 --dist loadgroup \
    --databricks-profile <profile> --catalog <catalog>
```

- Narrow it to a suite (`tests/suites/tools`), a file, or a test (`"...test_deploy.py::test_deploy[cli]"`).
- `-n 6 --dist loadgroup` runs tests in parallel; `loadgroup` keeps the dev tests on one worker,
  because local dev servers share a proxy port.
- Workspace resources the suite borrows but never creates are flags: `--catalog`,
  `--warehouse-id`, and `--genie-space-id` (without one the Genie tool is skipped).
- To set flags once for a shell or a CI job, use pytest's own `PYTEST_ADDOPTS`, e.g.
  `export PYTEST_ADDOPTS="--databricks-profile <profile> --catalog <catalog>"`.
- `--workspace fake` runs against no workspace; everything that needs one is skipped.
- `--keep-resources` keeps leftovers for debugging; each test still deletes its own resources.

The run id and output directory (logs, generated projects, `apps.jsonl`) are printed at the start:
`Agent Bricks E2E output: /tmp/agentbricks-e2e-<id> (run <id>)`.

## Cleanup

Every resource a run creates carries its run id, and is deleted at three points:

1. **When each test ends**, by the fixture that made it: `project` deletes the test's App, its stores,
   and everything its deploys created (`harness.cleanup.delete_app_resources`); `scratch_schema`
   deletes the test's schema.
2. **When the run ends**, `pytest_sessionfinish` deletes whatever a crashed test or a failed teardown
   left (`harness.cleanup.delete_run_leftovers`). A failure there fails the run.
3. **After a run that was killed** before step 2, for example by a CI timeout:

   ```bash
   uv run pytest tests/suites --cleanup-only --run-id <id> --databricks-profile <profile> --catalog <catalog>
   ```

   CI passes its own `--run-id` and runs this in an `always()` step.

## Layout

| Path | What it is |
|---|---|
| `conftest.py` | All pytest wiring: options, hooks, and the fixtures every suite shares |
| `harness/run.py` | `RunConfig`, the run id, and the name prefixes every resource carries |
| `harness/target.py` | The `--workspace live\|fake` seam the CLI and fixtures talk to |
| `harness/cli.py` | Runs `agentbricks` commands: `cli.run("tools add ...")`, and `cli.start("dev ...")` in the background |
| `harness/agent_endpoint.py` | `AgentEndpoint`: a running dev server or deployed App, and invoking it over HTTP |
| `harness/project.py` | A test's project: run-scoped path and App name, its files, and `pin_runtime` |
| `harness/workspace.py` | Databricks SDK helpers for what is not under test: SQL, UC, Apps, grants |
| `harness/cleanup.py` | Deleting what the suites create, per App and per run |
| `harness/logs.py` | Test-log output; every command is echoed so a run's log replays it |
| `<suite>/` | One folder per feature: `tools`, `memory_store`, `session_store`, `tracing` |

`conftest.py` files are the only code pytest calls by name; everything in `harness/` is plain Python
that you can follow through its imports.

## Adding things

- **A tool:** add one `ToolDefinition(name, requires, build)` in `tools/tool_definitions.py`.
  `requires` names the fixtures it needs; tools whose fixtures skip are left out and logged.
- **A suite:** add a folder with `test_*.py` files. A test takes the `project` fixture, runs
  `agentbricks_cli.run(f"init ... {project.path}")`, then `pin_runtime(project, ...)` so the project
  runs the checkout's runtime; teardown fails a test that skips the pin. Add a `conftest.py` only for resources the CLI
  under test does not create itself; create them per test and delete them in the fixture's teardown.
