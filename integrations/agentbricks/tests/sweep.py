"""Delete whatever one E2E run leaked, inside pytest or as a separate step after it.

Per-test teardown deletes what each test created; this catches what a crashed test or a failed
teardown left behind. pytest runs it at the end of a run, but a job timeout or cancellation kills
pytest before that, which is exactly when leaks happen. So CI picks the run id up front and runs
this again in an ``always()`` step, from ``integrations/agentbricks``:

    uv run pytest tests/suites/tools --run-id "$RUN_ID" ...
    uv run python tests/sweep.py --run-id "$RUN_ID" --databricks-profile <profile>

Everything is found by the run's name prefix (Apps, source folders, schemas) or by the App
identities workers recorded in ``<output>/apps.jsonl`` (Lakebase databases and roles, which outlive
their App), so it touches nothing outside the run and is safe to run twice.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from common import (
    APP_PREFIX,
    RUN_ID_PATTERN,
    log,
    project_prefix,
    run_output_dir,
    scratch_schema_prefix,
)
from workspace_client import (
    Workspace,
    cleanup_app,
    delete_lakebase_leftovers,
    recorded_app_identities,
)

DEFAULT_CATALOG = os.environ.get("AGENTBRICKS_E2E_CATALOG", "main")


def sweep_run(workspace: Workspace, run_id: str, output: pathlib.Path, catalog: str) -> list[str]:
    """Delete everything the run left; returns a description of each failure."""
    failures: list[str] = []
    sweeps = (
        ("leaked Apps", lambda: _sweep_apps(workspace, run_id, output)),
        ("leaked Lakebase databases and roles", lambda: _sweep_lakebase(workspace, output)),
        ("leaked source folders", lambda: _sweep_deployment_sources(workspace, run_id)),
        ("leaked schemas", lambda: _sweep_schemas(workspace, catalog, run_id)),
    )
    for label, action in sweeps:
        try:
            failures.extend(action())
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    return failures


def _app_prefix(run_id: str) -> str:
    return f"{APP_PREFIX}{project_prefix(run_id)}"


def _sweep_apps(workspace: Workspace, run_id: str, output: pathlib.Path) -> list[str]:
    """Delete Apps whose per-test cleanup failed or never ran, found by the run's name prefix."""
    prefix = _app_prefix(run_id)
    recorded = recorded_app_identities(output)
    failures: list[str] = []
    for app in list(workspace.client.apps.list()):
        if app.name and app.name.startswith(prefix):
            log(f"sweeping leaked App {app.name}")
            failures += cleanup_app(
                workspace, app.name, runtime_store=True, identity=recorded.get(app.name)
            )
    return failures


def _sweep_lakebase(workspace: Workspace, output: pathlib.Path) -> list[str]:
    """Delete runtime databases and roles of this run's Apps, whether or not the App still exists."""
    recorded = list(recorded_app_identities(output).values())
    failures: list[str] = []
    for branch in sorted({item.branch for item in recorded if item.branch}):
        failures += delete_lakebase_leftovers(
            workspace,
            branch,
            # A Runtime Store create that fails partway records no branch but can still leave the
            # App's role behind, so look for branchless identities on every branch this run used.
            principals=[
                item.service_principal
                for item in recorded
                if item.branch in (branch, None) and item.service_principal
            ],
            database_ids=[
                item.database_id for item in recorded if item.branch == branch and item.database_id
            ],
        )
    return failures


def _sweep_deployment_sources(workspace: Workspace, run_id: str) -> list[str]:
    """Delete this run's synced source folders, which `apps delete` leaves behind."""
    failures: list[str] = []
    for path in workspace.deployment_sources_with_prefix(_app_prefix(run_id)):
        log(f"sweeping deployment source {path}")
        try:
            workspace.delete_workspace_path(path)
        except Exception as exc:
            failures.append(f"deployment source {path}: {exc}")
    return failures


def _sweep_schemas(workspace: Workspace, catalog: str, run_id: str) -> list[str]:
    """Delete this run's temporary schemas whose test teardown did not run, found by name prefix."""
    failures: list[str] = []
    for schema in workspace.schemas_with_prefix(catalog, scratch_schema_prefix(run_id)):
        log(f"sweeping leaked schema {schema}")
        try:
            workspace.delete_schema(schema)
        except Exception as exc:
            failures.append(f"schema {schema}: {exc}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete whatever one Agent Bricks E2E run leaked.")
    parser.add_argument("--run-id", required=True, help="The run id the pytest run was given.")
    parser.add_argument("--databricks-profile", help="Databricks CLI profile; default is ambient.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help="Catalog of the run's schemas.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="The run's output dir, for apps.jsonl (default: the one pytest derives from the id).",
    )
    args = parser.parse_args(argv)
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must be 1-8 lowercase letters or digits")
    output = args.output or run_output_dir(args.run_id)
    if not (output / "apps.jsonl").exists():
        # Without the record, Lakebase leftovers of Apps that are already gone can't be found.
        log(f"no apps.jsonl in {output}; sweeping only what the run prefix finds")
    failures = sweep_run(Workspace(args.databricks_profile), args.run_id, output, args.catalog)
    for failure in failures:
        log(f"cleanup warning | {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
