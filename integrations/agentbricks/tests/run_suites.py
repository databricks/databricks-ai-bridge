#!/usr/bin/env python3
"""Provision a suite's workspace dependencies, run its pytest tests, and tear everything down.

One entry point for local and CI runs. Auth comes from the environment (ambient Databricks
credentials, or DATABRICKS_CONFIG_PROFILE locally):

    uv run python tests/run_suites.py --profile nightly

Each ``matrices/<matrix>/suite.yaml`` manifest declares the dependencies (``requires``) its pytest directory
needs. Three kinds exist: ``existing`` (validated, never created or deleted), ``dab`` (a bundle
deployed before and destroyed after the tests), and ``uc_function`` (imperative SQL inside a
``dab`` schema, swept by the schema's destroy). The runner hands the resolved values to pytest
through the JSON file named by ``AGENTBRICKS_TEST_INPUTS`` and owns the evidence file.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections.abc import Sequence
from typing import Any

import yaml
from agentbricks_cli import delete_store
from common import (
    AUTHORING_PATHS,
    FRAMEWORKS,
    INPUTS_ENV,
    MATRIX_CELLS,
    MatrixError,
    Transcript,
    last_nonempty_line,
    now,
)
from databricks.sdk import WorkspaceClient
from workspace_client import Workspace, cleanup_app

TESTS_DIR = pathlib.Path(__file__).resolve().parent
MATRICES_DIR = TESTS_DIR / "matrices"
PACKAGE_DIR = TESTS_DIR.parent
REPOSITORY_ROOT = TESTS_DIR.parents[2]
PROFILES = ("ci", "nightly")
REQUIREMENT_KINDS = ("existing", "dab", "uc_function")
# Time held back from the suite timeout so teardown can finish after pytest is stopped.
TEARDOWN_RESERVE_SECONDS = 180

_WHEEL_SOURCE_FILES = (
    "databricks_agentkit/_api_client.py",
    "databricks_agentbricks/tool_access.py",
    "databricks_agentbricks/app_resources.py",
    "databricks_agentbricks/cli/deploy.py",
)
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@dataclasses.dataclass
class Suite:
    name: str
    path: pathlib.Path
    tests: list[str]
    runs: dict[str, dict[str, Any]]
    requires: list[dict[str, Any]]


@dataclasses.dataclass
class Options:
    run_profile: str
    output: pathlib.Path
    wheel: pathlib.Path | None
    commit_sha: str | None
    source_root: pathlib.Path
    genie_space_id: str | None
    bridge_sha: str | None
    warehouse_id: str | None
    databricks_profile: str | None
    app_auth_profile: str | None
    preprovisioned_app_catalog_access: bool
    keep_resources: bool
    pytest_args: list[str] = dataclasses.field(default_factory=list)


class Shell:
    """Subprocess and Databricks CLI wrappers that log every command."""

    def __init__(self, profile: str | None, output: pathlib.Path):
        self.profile = profile
        self.output = output
        self.transcript = Transcript(output / "commands.log")
        self.warehouse_id: str | None = None

    def profile_args(self) -> list[str]:
        return ["--profile", self.profile] if self.profile else []

    def auth_label(self) -> str:
        return f"profile {self.profile!r}" if self.profile else "ambient environment credentials"

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 300,
        log: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if log:
            self.transcript.command(argv, cwd)
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if log and result.stdout.strip():
            self.transcript.write(result.stdout)
        if log and result.stderr.strip():
            self.transcript.write(result.stderr)
        if check and result.returncode != 0:
            raise MatrixError(
                f"Command failed ({result.returncode}): {shlex.join(list(argv))}\n"
                f"{result.stderr or result.stdout}"
            )
        return result

    def run_long(
        self,
        label: str,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 1800,
    ) -> str:
        self.transcript.command(argv, cwd)
        log_path = self.output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_tick = 60.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    raise MatrixError(f"{label} timed out after {timeout:.0f}s; log: {log_path}")
                if elapsed >= next_tick:
                    self.transcript.write(
                        f"tick {now():%H:%M} | {label} | running | {last_nonempty_line(log_path)}"
                    )
                    next_tick += 60.0
                time.sleep(2)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        self.transcript.write(output)
        if process.returncode != 0:
            raise MatrixError(f"{label} failed ({process.returncode}); log: {log_path}")
        self.transcript.write(f"tick {now():%H:%M} | {label} | success")
        return output

    def databricks(self, args: Sequence[str], *, timeout: float = 300) -> Any:
        result = self.run(
            ["databricks", *args, *self.profile_args(), "--output", "json"],
            timeout=timeout,
        )
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise MatrixError(f"Databricks CLI returned invalid JSON: {result.stdout}") from exc


class Toolchain(Shell):
    """The wheel under test, its venv, and the workspace identity shared by every suite."""

    def __init__(self, options: Options):
        super().__init__(options.databricks_profile, options.output)
        self.options = options
        self.app_auth_profile = options.app_auth_profile or options.databricks_profile
        self.wheel = options.wheel.resolve() if options.wheel else None
        self.commit_sha = options.commit_sha.lower() if options.commit_sha else None
        self.source_provenance: dict[str, Any] = {}
        self.versions: dict[str, str] = {}
        self.host: str | None = None
        self.runner_venv = options.output / "runner-venv"
        self.agentbricks = self.runner_venv / "bin" / "agentbricks"

    def _app_auth_label(self) -> str:
        return (
            f"App auth profile {self.app_auth_profile!r}"
            if self.app_auth_profile
            else "ambient environment credentials"
        )

    def bootstrap(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        source_root = self.options.source_root.resolve()
        if self.commit_sha is None:
            self.commit_sha = (
                self.run(["git", "rev-parse", "HEAD"], cwd=source_root).stdout.strip().lower()
            )
        if re.fullmatch(r"[0-9a-f]{40}", self.commit_sha) is None:
            raise MatrixError("--commit-sha must be a full 40-character hexadecimal Git SHA")
        if self.options.bridge_sha is not None and (
            re.fullmatch(r"[0-9a-f]{40}", self.options.bridge_sha) is None
        ):
            raise MatrixError("--bridge-sha must be a 40-character lowercase Git commit SHA.")
        if self.wheel is None:
            dist = self.output / "dist"
            self.run(
                ["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=PACKAGE_DIR, timeout=600
            )
            wheels = sorted(dist.glob("*.whl"))
            if not wheels:
                raise MatrixError("uv build produced no wheel")
            self.wheel = wheels[-1].resolve()
        self.source_provenance = _source_provenance(source_root, self.commit_sha, self.wheel)
        self.run(["uv", "venv", str(self.runner_venv)], timeout=300)
        self.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.runner_venv / "bin" / "python"),
                str(self.wheel),
            ],
            timeout=600,
        )
        self.run([str(self.agentbricks), "tools", "--help"])
        version_commands = {
            "agentbricks": [str(self.agentbricks), "--version"],
            "databricks": ["databricks", "version"],
            "uv": ["uv", "--version"],
            "python": [str(self.runner_venv / "bin" / "python"), "--version"],
        }
        for name, command in version_commands.items():
            result = self.run(command, log=False, check=False)
            version = (result.stdout or result.stderr).strip()
            if result.returncode != 0 or not version:
                raise MatrixError(f"Could not record {name} version: {version or 'no output'}")
            self.versions[name] = version
        workspace_client = WorkspaceClient(profile=self.profile)
        app_auth_client = WorkspaceClient(profile=self.app_auth_profile)
        if not workspace_client.config.host:
            raise MatrixError(f"Could not resolve a host from {self.auth_label()}.")
        if not app_auth_client.config.host:
            raise MatrixError(f"Could not resolve a host from {self._app_auth_label()}.")
        self.host = workspace_client.config.host.rstrip("/")
        app_auth_host = app_auth_client.config.host.rstrip("/")
        if app_auth_host != self.host:
            raise MatrixError(f"{self._app_auth_label()} targets {app_auth_host}, not {self.host}.")
        if app_auth_client.config.auth_type == "pat":
            raise MatrixError(
                f"{self._app_auth_label()} uses a PAT. "
                "Databricks Apps /api routes require OAuth; run `databricks auth login` "
                "for a profile on the same workspace."
            )
        if not app_auth_client.config.authenticate().get("Authorization"):
            raise MatrixError(f"Could not resolve credentials from {self._app_auth_label()}.")


class SuiteRunner(Shell):
    def __init__(self, suite: Suite, toolchain: Toolchain, options: Options):
        super().__init__(toolchain.profile, options.output / suite.name)
        self.suite = suite
        self.toolchain = toolchain
        self.options = options
        self.run_id = uuid.uuid4().hex[:8]
        self.run_suffix = self.run_id[:6]
        self.cleanup_required = not options.keep_resources
        self.cleanup_complete = False
        self.cleanup_results: list[dict[str, Any]] = []
        self.started_at = now().isoformat()
        self.ended_at: str | None = None
        self.resolved: dict[str, str] = {}
        self.catalog: str | None = None
        self.schema: str | None = None
        self.uc_function: str | None = None
        self.transitive_uc_function: str | None = None
        self.uc_volume: str | None = None
        self.volume_marker: str | None = None
        self.volume_file_path: str | None = None
        self.genie_space_id: str | None = None
        self.bundle_dir: pathlib.Path | None = None
        self.bundle_env: dict[str, str] = {}
        self.bundle_target = "nightly" if options.run_profile == "nightly" else "dev"
        self.inputs_path = self.output / "inputs.json"
        # Per-worker files written by common.Evidence; glob patterns merge them.
        self.rows_glob = "rows.*.jsonl"
        self.registry_glob = "registry.*.jsonl"
        self.cleanup_glob = "cleanup.*.jsonl"
        self.grant_dir = self.output / "grant-checks"
        self._workspace: Workspace | None = None

    @property
    def workspace(self) -> Workspace:
        if self._workspace is None:
            self._workspace = Workspace(self.profile, self.transcript)
        return self._workspace

    # Provisioning

    def provision(self) -> None:
        for requirement in self.suite.requires:
            kind = requirement.get("kind")
            if kind == "existing":
                self._provision_existing(requirement)
            elif kind == "dab":
                self._provision_dab(requirement)
            elif kind == "uc_function":
                self._provision_uc_function(requirement)
            else:
                raise MatrixError(f"Unknown requirement kind {kind!r} in {self.suite.name}.")
        self.genie_space_id = self.resolved.get("genie_space")

    def _provision_existing(self, requirement: dict[str, Any]) -> None:
        name = requirement["name"]
        value = _expand(str(requirement["value"]), self._overrides())
        pattern = requirement.get("pattern")
        if pattern and re.fullmatch(pattern, value) is None:
            raise MatrixError(f"Requirement {name!r} value {value!r} does not match {pattern!r}.")
        check = requirement.get("check")
        if check:
            self.databricks([*check, value])
        self.resolved[name] = value
        if name == "catalog":
            self.catalog = value

    def _provision_dab(self, requirement: dict[str, Any]) -> None:
        name = requirement["name"]
        catalog = self.resolved.get(requirement["catalog"])
        if not catalog:
            raise MatrixError(f"Requirement {name!r} needs {requirement['catalog']!r} first.")
        self.bundle_dir = (self.suite.path.parent / requirement["bundle"]).resolve()
        self.bundle_env = {
            **os.environ,
            "BUNDLE_VAR_catalog_name": catalog,
            "BUNDLE_VAR_run_id": self.run_id,
        }
        bundle_args = ["-t", self.bundle_target, *self.profile_args()]
        self.run_long(
            "bundle-deploy",
            ["databricks", "bundle", "deploy", *bundle_args],
            cwd=self.bundle_dir,
            env=self.bundle_env,
            timeout=900,
        )
        summary = json.loads(
            self.run(
                ["databricks", "bundle", "summary", *bundle_args, "--output", "json"],
                cwd=self.bundle_dir,
                env=self.bundle_env,
                timeout=120,
            ).stdout
        )
        resource = summary.get("resources", {}).get("schemas", {}).get(name)
        if not isinstance(resource, dict):
            raise MatrixError(f"bundle summary has no schemas.{name}: {summary}")
        resolved_id = str(resource.get("id") or "")
        if resolved_id.count(".") == 1:
            schema = resolved_id
        else:
            schema = f"{resource.get('catalog_name') or catalog}.{resource.get('name')}"
        if schema.count(".") != 1 or "None" in schema:
            raise MatrixError(f"Could not resolve the deployed schema from {resource}.")
        self.schema = schema
        self.resolved[name] = schema

    def _provision_uc_function(self, requirement: dict[str, Any]) -> None:
        schema = self.resolved.get(requirement["schema"])
        if not schema:
            raise MatrixError(f"Requirement {requirement['name']!r} needs a deployed schema first.")
        self.warehouse_id = self.workspace.start_warehouse(self.options.warehouse_id)
        catalog, _, schema_name = schema.partition(".")
        suffix = uuid.uuid4().hex[:8]
        nested_function_name = f"agentbricks_nested_{suffix}"
        self.transitive_uc_function = f"{catalog}.{schema_name}.{nested_function_name}"
        self.workspace.sql(
            f"CREATE OR REPLACE FUNCTION `{catalog}`.`{schema_name}`.`{nested_function_name}`"
            "(value STRING) RETURNS STRING "
            "COMMENT 'Transitive Agent Bricks E2E marker; never declared in agent.toml' "
            "RETURN concat('AGENTBRICKS_UC_OK:', value)"
        )
        # Leave room for catalog and schema in the 64-character MCP tool name.
        function_name = f"ab_uc_{suffix}"
        self.uc_function = f"{catalog}.{schema_name}.{function_name}"
        volume_name = f"agentbricks_volume_{suffix}"
        self.uc_volume = f"{catalog}.{schema_name}.{volume_name}"
        self.workspace.sql(f"CREATE VOLUME `{catalog}`.`{schema_name}`.`{volume_name}`")
        self.volume_marker = f"AGENTBRICKS_VOLUME_{uuid.uuid4().hex}"
        self.volume_file_path = f"/Volumes/{catalog}/{schema_name}/{volume_name}/marker.txt"
        self.workspace.upload(self.volume_file_path, self.volume_marker.encode())
        exposed_tool_name = self.uc_function.replace(".", "__")
        if len(exposed_tool_name) > 64:
            raise MatrixError(
                "The UC function's MCP tool name would exceed 64 characters: "
                f"{exposed_tool_name!r}. Use a shorter catalog or run id."
            )
        self.workspace.sql(
            f"CREATE OR REPLACE FUNCTION `{catalog}`.`{schema_name}`.`{function_name}`"
            "(value STRING) RETURNS STRING "
            "COMMENT 'Deterministic Agent Bricks E2E marker tool' "
            f"RETURN `{catalog}`.`{schema_name}`.`{nested_function_name}`(value)"
        )
        self.resolved[requirement["name"]] = self.uc_function

    def _overrides(self) -> dict[str, str]:
        return (
            {"AGENTBRICKS_E2E_GENIE_SPACE_ID": self.options.genie_space_id}
            if self.options.genie_space_id
            else {}
        )

    def write_inputs(self) -> None:
        toolchain = self.toolchain
        inputs = {
            "suite": self.suite.name,
            "run_profile": self.options.run_profile,
            "databricks_profile": self.profile,
            "app_auth_profile": toolchain.app_auth_profile,
            "preprovisioned_app_catalog_access": self.options.preprovisioned_app_catalog_access,
            "bridge_sha": self.options.bridge_sha,
            "wheel": str(toolchain.wheel),
            "agentbricks_bin": str(toolchain.agentbricks),
            "output_dir": str(self.output),
            "run_id": self.run_id,
            "run_suffix": self.run_suffix,
            "host": toolchain.host,
            "catalog": self.catalog,
            "schema": self.schema,
            "uc_function": self.uc_function,
            "transitive_uc_function": self.transitive_uc_function,
            "uc_volume": self.uc_volume,
            "volume_marker": self.volume_marker,
            "volume_file_path": self.volume_file_path,
            "genie_space_id": self.genie_space_id,
            "warehouse_id": self.warehouse_id,
            "requirements": self.resolved,
        }
        self.inputs_path.parent.mkdir(parents=True, exist_ok=True)
        self.inputs_path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        self.inputs_path.chmod(0o600)

    # Test execution

    def _parallel_args(self) -> list[str]:
        # loadgroup keeps tests sharing an xdist_group (the local dev runs) on one worker.
        workers = int(self.suite.runs[self.options.run_profile].get("workers", 1))
        return ["-n", str(workers), "--dist", "loadgroup"] if workers > 1 else []

    def run_pytest(self, budget_seconds: float) -> int:
        tests = [str(TESTS_DIR / relative) for relative in self.suite.tests]
        argv = [
            sys.executable,
            "-m",
            "pytest",
            *tests,
            "-v",
            "-ra",
            "--capture=tee-sys",
            "-p",
            "no:cacheprovider",
            "--junitxml",
            str(self.output / "junit.xml"),
            *self._parallel_args(),
            *self.options.pytest_args,
        ]
        env = {**os.environ, INPUTS_ENV: str(self.inputs_path)}
        if self.options.databricks_profile:
            # The CLI auth token lookup is by host; pin the profile when several share one host.
            env["DATABRICKS_CONFIG_PROFILE"] = self.options.databricks_profile
        self.transcript.command(argv)
        process = subprocess.Popen(argv, cwd=TESTS_DIR, env=env, start_new_session=True)
        try:
            return process.wait(timeout=budget_seconds)
        except subprocess.TimeoutExpired:
            self.transcript.write(f"pytest exceeded its {budget_seconds:.0f}s budget; stopping it.")
            # SIGINT lets pytest unwind its fixtures (the dev servers) before teardown starts.
            _signal_group(process, signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                process.wait()
            return 124

    # Teardown

    def _registry(self) -> tuple[list[str], dict[str, dict[str, Any]], set[str]]:
        apps: list[str] = []
        projects: dict[str, dict[str, Any]] = {}
        cleaned: set[str] = set()
        for event in self._read_jsonl(self.registry_glob):
            name = event["app_name"]
            if event["kind"] == "app" and name not in apps:
                apps.append(name)
            elif event["kind"] == "project":
                projects.setdefault(name, {}).update(event)
            elif event["kind"] == "cleaned":
                cleaned.add(name)
        return apps, projects, cleaned

    def teardown(self) -> None:
        # Each test deletes its own App and stores; this sweeps whatever a test could not, such as
        # after a crash, and folds the tests' own cleanup results into the evidence.
        apps, projects, cleaned = self._registry()
        self.cleanup_results.extend(self._read_jsonl(self.cleanup_glob))
        for name in dict.fromkeys([*apps, *projects]):
            if name not in cleaned:
                self.cleanup_results.extend(
                    cleanup_app(
                        self.workspace,
                        functools.partial(
                            delete_store,
                            self.toolchain.agentbricks,
                            self.profile,
                            self.transcript,
                        ),
                        name,
                        projects.get(name, {}),
                        has_app=name in apps,
                    )
                )
        for label, function in (
            ("UC function", self.uc_function),
            ("transitive UC function", self.transitive_uc_function),
        ):
            if not function:
                continue
            catalog, schema, function_name = function.split(".")
            try:
                self.workspace.sql(
                    f"DROP FUNCTION IF EXISTS `{catalog}`.`{schema}`.`{function_name}`"
                )
                self.cleanup_results.append(
                    {"resource": f"function:{function}", "status": "deleted"}
                )
            except Exception as exc:
                self.transcript.write(f"cleanup warning | {label} | {exc}")
                self.cleanup_results.append(
                    {"resource": f"function:{function}", "status": "failed", "detail": str(exc)}
                )
        if self.uc_volume:
            catalog, schema, volume_name = self.uc_volume.split(".")
            if self.volume_file_path:
                try:
                    self.workspace.delete_file(self.volume_file_path)
                    self.cleanup_results.append(
                        {"resource": f"file:{self.volume_file_path}", "status": "deleted"}
                    )
                except Exception as exc:
                    self.transcript.write(f"cleanup warning | UC volume file | {exc}")
                    self.cleanup_results.append(
                        {
                            "resource": f"file:{self.volume_file_path}",
                            "status": "failed",
                            "detail": str(exc),
                        }
                    )
            try:
                self.workspace.sql(f"DROP VOLUME IF EXISTS `{catalog}`.`{schema}`.`{volume_name}`")
                self.cleanup_results.append(
                    {"resource": f"volume:{self.uc_volume}", "status": "deleted"}
                )
            except Exception as exc:
                self.transcript.write(f"cleanup warning | UC volume | {exc}")
                self.cleanup_results.append(
                    {
                        "resource": f"volume:{self.uc_volume}",
                        "status": "failed",
                        "detail": str(exc),
                    }
                )
        if self.bundle_dir is not None:
            resource_name = f"bundle:{self.schema or self.suite.name}"
            try:
                self.run_long(
                    "bundle-destroy",
                    [
                        "databricks",
                        "bundle",
                        "destroy",
                        "--auto-approve",
                        "-t",
                        self.bundle_target,
                        *self.profile_args(),
                    ],
                    cwd=self.bundle_dir,
                    env=self.bundle_env,
                    timeout=900,
                )
                self.cleanup_results.append({"resource": resource_name, "status": "deleted"})
            except Exception as exc:
                self.transcript.write(f"cleanup warning | bundle destroy | {exc}")
                self.cleanup_results.append(
                    {"resource": resource_name, "status": "failed", "detail": str(exc)}
                )
        self.cleanup_complete = not any(
            result.get("status") == "failed" for result in self.cleanup_results
        )

    # Evidence

    def _read_jsonl(self, pattern: str) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for path in sorted(self.output.glob(pattern))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _grant_checks(self) -> list[dict[str, Any]]:
        checks = []
        for path in sorted(self.grant_dir.glob("*.json")) if self.grant_dir.exists() else ():
            recorded = json.loads(path.read_text(encoding="utf-8"))
            checks.append(
                {
                    "app_name": recorded.get("app_name"),
                    "authoring": recorded.get("authoring"),
                    "service_principal_client_id": recorded.get("service_principal_client_id"),
                    "initial": recorded.get("initial", {}),
                    "post_manual_transitive_grant": recorded.get(
                        "post_manual_transitive_grant", {}
                    ),
                    "manual_transitive_grant_applied": recorded.get(
                        "manual_transitive_grant_applied", False
                    ),
                }
            )
        return checks

    def write_evidence(self) -> pathlib.Path:
        toolchain = self.toolchain
        wheel = toolchain.wheel
        payload = {
            "schema_version": 1,
            "suite": self.suite.name,
            "run_profile": self.options.run_profile,
            "run_id": self.run_id,
            "commit_sha": toolchain.commit_sha,
            "source_provenance": toolchain.source_provenance,
            "template_repo": "wheel://databricks-agentbricks",
            "template_ref": _sha256(wheel) if wheel else None,
            "workspace_host": toolchain.host,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "versions": toolchain.versions,
            "profile": self.profile,
            "app_auth_profile": toolchain.app_auth_profile,
            "bridge_sha": self.options.bridge_sha,
            "preprovisioned_app_catalog_access": self.options.preprovisioned_app_catalog_access,
            "generated_at": now().isoformat(),
            "wheel": str(wheel) if wheel else None,
            "wheel_sha256": _sha256(wheel) if wheel else None,
            "schema": self.schema,
            "uc_function": self.uc_function,
            "transitive_uc_function": self.transitive_uc_function,
            "uc_volume": self.uc_volume,
            "volume_marker": self.volume_marker,
            "volume_file_path": self.volume_file_path,
            "genie_space_id": self.genie_space_id,
            "warehouse_id": self.warehouse_id,
            "grant_checks": self._grant_checks(),
            "cleanup_required": self.cleanup_required,
            "cleanup_complete": self.cleanup_complete,
            "cleanup": self.cleanup_results,
            "rows": self._read_jsonl(self.rows_glob),
        }
        target = self.output / "evidence.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return target

    # Orchestration

    def execute(self) -> int:
        timeout_minutes = self.suite.runs[self.options.run_profile]["timeout_minutes"]
        started = time.monotonic()
        pytest_returncode: int | None = None
        try:
            self.provision()
            self.write_inputs()
            remaining = (
                timeout_minutes * 60 - TEARDOWN_RESERVE_SECONDS - (time.monotonic() - started)
            )
            pytest_returncode = self.run_pytest(max(60.0, remaining))
        except Exception as exc:
            self.transcript.write(f"suite {self.suite.name} failed before tests finished: {exc}")
        finally:
            if self.cleanup_required:
                self.teardown()
            else:
                self.transcript.write("Resources retained (--keep-resources); skipping teardown.")
            self.ended_at = now().isoformat()
            evidence = self.write_evidence()
        # A filtered run covers only some cells, so the full-matrix check would always fail.
        verified = 0 if self.options.pytest_args else verify_evidence(evidence)
        return 0 if pytest_returncode == 0 and verified == 0 else 1


def _signal_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _expand(value: str, overrides: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = overrides.get(name) or os.environ.get(name) or default
        if not resolved:
            raise MatrixError(f"Required environment variable {name} is not set.")
        return resolved

    return _ENV_REFERENCE.sub(replace, value)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(
    source_root: pathlib.Path, commit_sha: str, wheel: pathlib.Path
) -> dict[str, Any]:
    def git(*args: str, text: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args],
            cwd=source_root,
            capture_output=True,
            text=text,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr if text else result.stderr.decode(errors="replace")
            raise MatrixError(f"Could not inspect source provenance: {detail.strip()}")
        return result

    head = git("rev-parse", "HEAD").stdout.strip().lower()
    if head != commit_sha.lower():
        raise MatrixError(
            f"Claimed commit {commit_sha} does not match source checkout HEAD {head}."
        )
    status = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    source_hashes: dict[str, str] = {}
    try:
        with zipfile.ZipFile(wheel) as archive:
            for member in _WHEEL_SOURCE_FILES:
                source_path = source_root / "integrations" / "agentbricks" / "src" / member
                source_bytes = source_path.read_bytes()
                try:
                    wheel_bytes = archive.read(member)
                except KeyError as exc:
                    raise MatrixError(f"Built wheel is missing source module {member}.") from exc
                if wheel_bytes != source_bytes:
                    raise MatrixError(
                        f"Built wheel module {member} does not match source checkout."
                    )
                source_hashes[member] = hashlib.sha256(wheel_bytes).hexdigest()
    except zipfile.BadZipFile as exc:
        raise MatrixError(f"Built wheel is not a readable zip archive: {wheel}") from exc
    diff = git("diff", "--binary", "HEAD", text=False).stdout
    return {
        "source_head_sha": head,
        "source_dirty": bool(status),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "wheel_source_matches": True,
        "wheel_source_sha256": source_hashes,
    }


def _verify_tools_evidence(document: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    """Sandbox-marker and grant proofs, which only the tools matrix produces."""
    grant_checks = document.get("grant_checks", [])
    sandbox_markers = {
        "sandbox": document.get("volume_marker"),
    }
    volume_file_path = document.get("volume_file_path")
    if (
        any(not isinstance(marker, str) or not marker for marker in sandbox_markers.values())
        or not isinstance(volume_file_path, str)
        or not volume_file_path.startswith("/Volumes/")
    ):
        sys.stdout.write("sandbox evidence: hidden marker provenance is missing or invalid\n")
        return 1
    for tool_kind, marker in sandbox_markers.items():
        if not isinstance(marker, str):
            continue  # unreachable: validated above
        matching_rows = [row for row in rows if row.get("tool_kind") == tool_kind]
        if any(
            row.get("expected") != marker or marker not in str(row.get("actual", ""))
            for row in matching_rows
        ):
            sys.stdout.write(
                f"sandbox evidence: exact hidden marker proof failed for {tool_kind}\n"
            )
            return 1
    # One deploy per authoring path proves the transitive exclusion and the manual grant. Repeat-deploy
    # idempotency is proven by the incremental CLI deploy rows, which re-check earlier tools' grants.
    transitive = [check for check in grant_checks if check.get("manual_transitive_grant_applied")]
    expected_transitive = len(FRAMEWORKS) * len(AUTHORING_PATHS)
    if len(transitive) != expected_transitive or any(
        "EXECUTE" in check.get("initial", {}).get("transitive_direct_privileges", [])
        or "EXECUTE" in check.get("initial", {}).get("transitive_effective_privileges", [])
        or "EXECUTE"
        not in check.get("post_manual_transitive_grant", {}).get("direct_privileges", [])
        or "EXECUTE"
        not in check.get("post_manual_transitive_grant", {}).get("effective_privileges", [])
        for check in transitive
    ):
        sys.stdout.write(
            f"grant evidence: expected {expected_transitive} transitive exclusion/manual-grant "
            f"proofs, got {len(transitive)} (or one failed)\n"
        )
        return 1
    return 0


def verify_evidence(path: pathlib.Path, *, require_cleanup: bool = True) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    provenance_fields = ("commit_sha", "workspace_host", "started_at", "ended_at")
    missing_provenance = [field for field in provenance_fields if not document.get(field)]
    template_repo = document.get("template_repo")
    template_ref = document.get("template_ref")
    if (
        not isinstance(template_repo, str)
        or not template_repo
        or not isinstance(template_ref, str)
        or not template_ref
    ):
        missing_provenance.append("template_repo/template_ref")
    elif template_repo.startswith("wheel://") and template_ref != document.get("wheel_sha256"):
        missing_provenance.append("wheel_template_ref")
    source_provenance = document.get("source_provenance")
    if (
        not isinstance(source_provenance, dict)
        or source_provenance.get("source_head_sha") != document.get("commit_sha")
        or source_provenance.get("wheel_source_matches") is not True
        or not isinstance(source_provenance.get("wheel_source_sha256"), dict)
        or not source_provenance.get("wheel_source_sha256")
    ):
        missing_provenance.append("source_provenance")
    versions = document.get("versions")
    if not isinstance(versions, dict) or any(
        not versions.get(name) for name in ("agentbricks", "databricks", "uv", "python")
    ):
        missing_provenance.append("versions")
    if missing_provenance:
        sys.stdout.write(f"evidence provenance missing: {sorted(missing_provenance)}\n")
        return 1
    suite = document.get("suite")
    cells = MATRIX_CELLS.get(suite) if isinstance(suite, str) else None
    if cells is None:
        sys.stdout.write(
            f"evidence suite {suite!r} is not a known matrix: {sorted(MATRIX_CELLS)}\n"
        )
        return 1
    rows = document.get("rows", [])
    expected = {
        (framework, authoring, runtime, tool)
        for framework in FRAMEWORKS
        for authoring, runtime, tool in cells
    }
    actual = {
        (row["framework"], row["authoring"], row["runtime"], row["tool_kind"]) for row in rows
    }
    duplicates = len(rows) - len(actual)
    passed = sum(row.get("status") == "pass" for row in rows)
    failed = sum(row.get("status") == "fail" for row in rows)
    skipped = len(expected - actual)
    sys.stdout.write(f"{passed} passed, {failed} failed, {skipped} skipped\n")
    if actual != expected or duplicates or passed != len(expected):
        if expected - actual:
            sys.stdout.write(f"missing cells: {sorted(expected - actual)}\n")
        if duplicates:
            sys.stdout.write(f"duplicate rows: {duplicates}\n")
        return 1
    if suite == "tools" and _verify_tools_evidence(document, rows):
        return 1
    cleanup_required = document.get("cleanup_required")
    if not isinstance(cleanup_required, bool):
        sys.stdout.write("cleanup evidence: cleanup_required was not recorded\n")
        return 1
    cleanup = document.get("cleanup", [])
    if (
        require_cleanup
        and cleanup_required
        and (
            document.get("cleanup_complete") is not True
            or not isinstance(cleanup, list)
            or any(result.get("status") == "failed" for result in cleanup)
            or any(
                result.get("resource", "").startswith("app:")
                and result.get("status") == "deleted"
                and not result.get("confirmed_absent_at")
                for result in cleanup
            )
        )
    ):
        sys.stdout.write("cleanup evidence: required cleanup is incomplete or failed\n")
        if isinstance(cleanup, list):
            for result in cleanup:
                if result.get("status") == "failed":
                    sys.stdout.write(
                        f"cleanup failed | {result.get('resource')} | {result.get('detail')}\n"
                    )
                # teardown() always stamps confirmed_absent_at on deleted apps, so
                # this only fires for evidence from an older runner or a hand-edited file.
                elif (
                    result.get("resource", "").startswith("app:")
                    and result.get("status") == "deleted"
                    and not result.get("confirmed_absent_at")
                ):
                    sys.stdout.write(
                        f"cleanup unconfirmed | {result.get('resource')} |"
                        " missing confirmed_absent_at\n"
                    )
        return 1
    if suite == "tools":
        grant_checks = document.get("grant_checks", [])
        transitive = [c for c in grant_checks if c.get("manual_transitive_grant_applied")]
        sys.stdout.write(f"{len(transitive)} transitive grant proofs passed\n")
    return 0


def load_suites() -> list[Suite]:
    suites = []
    for path in sorted(MATRICES_DIR.glob("*/suite.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for requirement in document.get("requires", []):
            if requirement.get("kind") not in REQUIREMENT_KINDS:
                raise MatrixError(f"{path.parent.name}: unknown requirement kind in {requirement}.")
        suites.append(
            Suite(
                name=document["name"],
                path=path,
                tests=list(document["tests"]),
                runs=document.get("runs", {}),
                requires=list(document.get("requires", [])),
            )
        )
    return suites


def select_suites(profile: str, names: Sequence[str]) -> list[Suite]:
    available = load_suites()
    if names:
        by_name = {suite.name: suite for suite in available}
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise MatrixError(f"Unknown suite(s) {unknown}; available: {sorted(by_name)}")
        selected = [by_name[name] for name in dict.fromkeys(names)]
        missing = [suite.name for suite in selected if profile not in suite.runs]
        if missing:
            raise MatrixError(f"Suite(s) {missing} do not run under the {profile!r} profile.")
        return selected
    return [suite for suite in available if profile in suite.runs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, help="Run profile selecting suites.")
    parser.add_argument("--suite", action="append", default=[], help="Suite name; repeatable.")
    parser.add_argument(
        "--output", type=pathlib.Path, help="Output directory (default: a tempdir)."
    )
    parser.add_argument("--wheel", type=pathlib.Path, help="Prebuilt wheel (default: build one).")
    parser.add_argument("--commit-sha", help="Source commit (default: HEAD of --source-root).")
    parser.add_argument("--source-root", type=pathlib.Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--genie-space-id",
        help="Overrides AGENTBRICKS_E2E_GENIE_SPACE_ID for the genie_space requirement.",
    )
    parser.add_argument(
        "--bridge-sha", help="Immutable bridge commit for generated App dependencies."
    )
    parser.add_argument("--warehouse-id")
    parser.add_argument(
        "--databricks-profile",
        help="Databricks CLI profile; default is ambient credentials (DATABRICKS_CONFIG_PROFILE).",
    )
    parser.add_argument(
        "--app-auth-profile",
        help="OAuth profile for deployed App /api calls; defaults to --databricks-profile.",
    )
    parser.add_argument(
        "--preprovisioned-app-catalog-access",
        action="store_true",
        help="Skip per-App USE CATALOG grants because catalog access is pre-provisioned.",
    )
    parser.add_argument("--keep-resources", action="store_true")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra pytest argument, e.g. -k test_dev (repeatable). Skips the full-matrix evidence check.",
    )
    parser.add_argument("--verify-evidence", type=pathlib.Path)
    args = parser.parse_args()
    if args.verify_evidence is None and args.profile is None:
        parser.error("--profile is required unless --verify-evidence is used")
    return args


def main() -> int:
    args = parse_args()
    if args.verify_evidence:
        return verify_evidence(args.verify_evidence)
    try:
        suites = select_suites(args.profile, args.suite)
    except MatrixError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if not suites:
        sys.stderr.write(f"No suites run under the {args.profile!r} profile.\n")
        return 2
    output = args.output or pathlib.Path(tempfile.mkdtemp(prefix="agentbricks-suites-"))
    options = Options(
        run_profile=args.profile,
        output=output.expanduser().resolve(),
        wheel=args.wheel,
        commit_sha=args.commit_sha,
        source_root=args.source_root,
        genie_space_id=args.genie_space_id or os.environ.get("AGENTBRICKS_E2E_GENIE_SPACE_ID"),
        bridge_sha=args.bridge_sha,
        warehouse_id=args.warehouse_id,
        databricks_profile=args.databricks_profile,
        app_auth_profile=args.app_auth_profile,
        preprovisioned_app_catalog_access=args.preprovisioned_app_catalog_access,
        keep_resources=args.keep_resources,
        pytest_args=args.pytest_arg,
    )
    toolchain = Toolchain(options)
    try:
        toolchain.bootstrap()
    except MatrixError as exc:
        toolchain.transcript.write(f"bootstrap failed: {exc}")
        return 1
    failed = [
        suite.name for suite in suites if SuiteRunner(suite, toolchain, options).execute() != 0
    ]
    if failed:
        sys.stdout.write(f"suites failed: {failed}; output: {options.output}\n")
        return 1
    sys.stdout.write(f"suites passed: {[suite.name for suite in suites]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
