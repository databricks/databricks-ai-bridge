"""Deploy the AgentKit orchestrator bound to published source; no agent installation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import click
import requests
import tomlkit
from databricks.sdk.errors import NotFound
from databricks.sdk.service import apps

from databricks_agentbricks.clients import managed_runtime_store
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.prepared_environment import read_manifest, services
from databricks_agentbricks.services.environment_features import provision_features
from databricks_agentkit._api_client import _AgentBricksApiClient, _workspace_client
from databricks_agentkit.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_USERNAME_ENV,
)
from databricks_agentkit.sandbox.bundle import bundle_contents
from databricks_agentkit.sandbox.catalog import WorkspaceCatalog
from databricks_agentkit.sandbox.environment import (
    bind_environment,
    on_demand,
    sandbox_scope,
    saved_bundle,
)
from databricks_agentkit.sandbox.features import source_features


def stage_runtime(
    target, manifest, principal, *, index_url, sandbox_idle_timeout=900, sandbox_warm_pool_size=0
):
    """Bundle the installed editable SDK; never read the local agent project."""
    package = Path(__file__).resolve().parents[3]
    if not (package / "pyproject.toml").exists():
        raise AgentCliError(
            "This prototype requires an editable Agent Bricks installation to package the runtime."
        )
    # Apps can reuse an installed package when its version is unchanged, even
    # after the wheel at the same Workspace path has been overwritten. Build a
    # snapshot with a content-derived local version; leave the checkout intact.
    with tempfile.TemporaryDirectory(prefix="agentbricks-runtime-build-") as directory:
        project = Path(directory)
        shutil.copytree(
            package / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        for name in ("pyproject.toml", "README.md", "NOTICE"):
            if (package / name).is_file():
                shutil.copyfile(package / name, project / name)
        digest = hashlib.sha256()
        for file in sorted(project.rglob("*")):
            if file.is_file():
                digest.update(file.relative_to(project).as_posix().encode() + b"\0")
                digest.update(file.read_bytes() + b"\0")
        metadata = tomlkit.parse((project / "pyproject.toml").read_text())
        base_version = str(metadata["project"]["version"]).split("+", 1)[0]
        metadata["project"]["version"] = f"{base_version}+sandbox.{digest.hexdigest()[:16]}"
        (project / "pyproject.toml").write_text(tomlkit.dumps(metadata))
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--no-sources",
                "--out-dir",
                str(target.resolve()),
                str(project),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    wheel = next(target.glob("*.whl"))
    (target / "requirements.txt").write_text(f"./{wheel.name}[sandbox]\n")
    (target / "environment.json").write_text(json.dumps(manifest, indent=2))
    environment = {
        "SANDBOX_NAMESPACE": manifest["namespace"],
        "SANDBOX_SCHEMA": manifest["namespace"].replace("-", "_"),
        "SANDBOX_CATALOG_PATH": manifest.get("catalog_path")
        or f"/Shared/agentbricks-runtime/{manifest['namespace']}",
        "SANDBOX_TRANSPORT": "https",
        "SANDBOX_IDLE_TIMEOUT_SECONDS": str(sandbox_idle_timeout),
        "SANDBOX_WARM_POOL_SIZE": str(sandbox_warm_pool_size),
        "SANDBOX_WARM_POOL_GENERATION": str(time.time_ns()),
        "AGENTBRICKS_ENVIRONMENT_FILE": "environment.json",
        "PIP_INDEX_URL": index_url,
        "UV_DEFAULT_INDEX": index_url,
    }
    if store := manifest.get("runtime_store"):
        environment.update(
            {
                RUNTIME_STORE_LAKEBASE_BRANCH_ENV: store["branch"],
                RUNTIME_STORE_DATABASE_ENV: store["database"],
                RUNTIME_STORE_USERNAME_ENV: store["username"],
            }
        )
    else:
        lakebase = manifest["lakebase"]
        environment.update(
            {
                "LAKEBASE_ENDPOINT": lakebase["endpoint"],
                "PGHOST": lakebase["host"],
                "PGDATABASE": lakebase["database"],
                "PGUSER": principal,
            }
        )
    (target / "app.yaml").write_text(
        json.dumps(
            {
                "command": ["python", "-m", "databricks_agentkit.sandbox"],
                "env": [{"name": key, "value": value} for key, value in environment.items()],
            },
            indent=2,
        )
    )


def deploy_prepared(
    obj,
    name,
    environment,
    *,
    workspace_path=None,
    instances=None,
    sandbox_idle_timeout=900,
    sandbox_warm_pool_size=0,
    index_url="https://pypi.org/simple/",
):
    from databricks_agentbricks.services.deployment.names import DeploymentName, _prefixed_name

    name = DeploymentName(_prefixed_name(name))
    workspace = _workspace_client(obj.profile)
    manifest = read_manifest(workspace, environment)
    lazy = on_demand(manifest)
    managed = "runtime_store" in manifest
    if managed and (
        not lazy
        or manifest["runtime_store"] != {"name": f"runtime-stores/{name}", "owner_app": name}
    ):
        raise AgentCliError("Managed source publication belongs to a different App or is invalid.")
    if manifest.get("status") != ("published" if lazy else "ready"):
        raise AgentCliError(
            "Deployment requires a published or legacy ready environment.",
            hint="Publish source with agentbricks deploy APP --sandbox --source PROJECT.",
        )
    # Lakebox restricts lifecycle and exec to the sandbox's creator. Workspace
    # file grants and Lakebase grants do not confer sandbox access. Check before
    # creating, updating, or deploying an App; never copy personal credentials.
    try:
        app = workspace.apps.get(name)
    except NotFound:
        app = None
    principal = (manifest.get("execution_identity") or {}).get("principal")
    if not lazy and not principal:
        raise AgentCliError(
            "Environment has no recorded sandbox execution identity.",
            hint=f"Run agentbricks environment prepare {environment} to refresh its metadata.",
        )
    if not lazy and (app is None or principal != app.service_principal_client_id):
        target_identity = app.service_principal_client_id if app else "a new App service principal"
        raise AgentCliError(
            f"Prepared sandbox belongs to {principal}; deployment would run as {target_identity}.",
            hint="Lakebox currently restricts sandbox lifecycle and execution to its creator. "
            "Prepare under the target App's execution identity before deploying. "
            "For a user-prepared environment, use environment invoke directly or the local "
            "playground UI. No App settings or agent installation were changed.",
        )
    registry = None
    if managed:
        bundle = saved_bundle(WorkspaceCatalog(workspace, manifest["catalog_path"]), manifest)
    else:
        registry, control, _, orchestrator = services(workspace, manifest)
        bind_environment(orchestrator, manifest)
        bundle = saved_bundle(control, manifest)
    if lazy and not managed:
        owners = (
            registry.execution_principals()
            if sandbox_scope(manifest) == "session"
            else [registry.runtime(manifest["runtime_key"])["execution_principal"]]
        )
        for owner in owners:
            if owner and (app is None or owner != app.service_principal_client_id):
                raise AgentCliError(
                    f"Environment runtime belongs to {owner}; target App has a different execution identity.",
                    hint="Publish a new environment for this App. Each session's first invocation must create "
                    "its sandbox under the App service principal. No App settings were changed.",
                )
    elif not lazy:
        orchestrator.invoke(
            manifest["preparation_id"],
            "",
            {},
            bundle=bundle,
            entrypoint=manifest["entrypoint"],
            _prepare_only=True,
            _allow_install=False,
        )
    features = source_features(bundle_contents(bundle.data).get("agent.toml"))
    click.echo(f"Deploying orchestrator {name} for environment {manifest['name']}", err=True)
    resources = list(app.resources or [] if app else [])
    if not managed:
        resource = apps.AppResource(
            name="sandbox-registry",
            postgres=apps.AppResourcePostgres(
                branch=manifest["lakebase"]["endpoint"].split("/endpoints/")[0],
                database=manifest["lakebase"]["database_resource"],
                permission=apps.AppResourcePostgresPostgresPermission.CAN_CONNECT_AND_CREATE,
            ),
        )
        resources = [item for item in resources if item.name != resource.name] + [resource]
    desired = apps.App(name=name, resources=resources)
    mask = ["resources"]
    if instances is not None:
        desired.compute_min_instances = desired.compute_max_instances = instances
        mask.extend(["compute_min_instances", "compute_max_instances"])
    if app is None:
        app = workspace.apps.create(desired).result()
    else:
        workspace.apps.create_update(name, app=desired, update_mask=",".join(mask)).result()
        app = workspace.apps.get(name)
    deadline = time.monotonic() + 120
    while not app.service_principal_client_id:
        if time.monotonic() >= deadline:
            raise AgentCliError("App execution identity was not provisioned before the deadline.")
        time.sleep(2)
        app = workspace.apps.get(name)
    if managed:
        backend = managed_runtime_store.get_or_create_backend(
            _AgentBricksApiClient(workspace_client=workspace), name, app.service_principal_client_id
        )
        # Coordinates are deployment configuration. The immutable source manifest
        # only names its owner; the CLI never opens the App's SQL database.
        manifest = {
            **manifest,
            "runtime_store": {
                **manifest["runtime_store"],
                "branch": backend.branch,
                "database": backend.database_id,
                "username": backend.username,
            },
        }
    else:
        while True:
            with registry.connect() as db:
                role = db.execute(
                    "SELECT 1 FROM pg_roles WHERE rolname=%s", (app.service_principal_client_id,)
                ).fetchone()
            if role:
                break
            if time.monotonic() >= deadline:
                raise AgentCliError("App Lakebase role was not provisioned before the deadline.")
            time.sleep(2)
        registry.grant_app_access(app.service_principal_client_id)
    # Source bindings follow the agent into its sandbox. Provision managed
    # resources under the creator and grant the App identity used by sandbox RPCs.
    manifest = {
        **manifest,
        "workbench": provision_features(
            workspace, features, app.service_principal_client_id, profile=obj.profile
        ),
    }
    # Grant the App read access to this publication, not its parent user folder.
    from databricks.sdk.service.workspace import (
        WorkspaceObjectAccessControlRequest,
        WorkspaceObjectPermissionLevel,
    )

    folders = {str(Path(manifest["manifest_path"]).parent)}
    if manifest.get("catalog_path"):
        folders.add(manifest["catalog_path"])
    for folder_path in sorted(folders):
        folder = workspace.workspace.get_status(folder_path)
        workspace.workspace.update_permissions(
            "directories",
            str(folder.object_id),
            access_control_list=[
                WorkspaceObjectAccessControlRequest(
                    service_principal_name=app.service_principal_client_id,
                    permission_level=WorkspaceObjectPermissionLevel.CAN_READ,
                )
            ],
        )
    user = workspace.current_user.me().user_name
    remote = workspace_path or f"/Workspace/Users/{user}/agentbricks_deployments/{name}"
    # Workspace files accept /Users/...; the Apps deployment API requires /Workspace/....
    remote = "/Workspace" + remote.removeprefix("/Workspace")
    if managed:
        # Each rollout refers to a complete, isolated upload. Concurrent deploys
        # need no publisher SQL lock and cannot mix source/config/wheel files.
        remote = f"{remote}/rollouts/{uuid4().hex}"
    from databricks.sdk.service.workspace import ImportFormat

    with tempfile.TemporaryDirectory(prefix="agentbricks-orchestrator-") as directory:
        target = Path(directory)
        stage_runtime(
            target,
            manifest,
            app.service_principal_client_id,
            index_url=index_url,
            sandbox_idle_timeout=sandbox_idle_timeout,
            sandbox_warm_pool_size=sandbox_warm_pool_size,
        )
        workspace.workspace.mkdirs(remote)
        for file in target.iterdir():
            workspace.workspace.upload(
                f"{remote}/{file.name}",
                file.read_bytes(),
                format=ImportFormat.RAW,
                overwrite=not managed,
            )
    if app.compute_status is None or app.compute_status.state != apps.ComputeState.ACTIVE:
        workspace.apps.start(name).result()
    workspace.apps.deploy(name, apps.AppDeployment(source_code_path=remote)).result()
    app = workspace.apps.get(name)
    if not app.url:
        raise AgentCliError("Deployed App has no URL; inspect its deployment status.")
    deadline = time.monotonic() + 180
    last = "not checked"
    while time.monotonic() < deadline:
        try:
            reply = requests.get(
                app.url.rstrip("/") + "/health",
                headers=workspace.config.authenticate(),
                timeout=15,
                allow_redirects=False,
            )
            last = str(reply.status_code)
            if reply.status_code == 200:
                result = {
                    "deployment": name,
                    "url": app.url,
                    "environment": manifest["name"],
                    "manifest_path": manifest["manifest_path"],
                    "sandbox_warm_pool_size": sandbox_warm_pool_size,
                    "sandbox_id": (manifest.get("preparation") or {}).get("sandbox_id"),
                    "installation": "on session preparation or first invocation"
                    if lazy
                    else "prepared",
                    "source_hash": manifest["source_hash"],
                    **({"runtime_store": manifest["runtime_store"]["name"]} if managed else {}),
                }
                return result
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(2)
    raise AgentCliError(
        "Orchestrator deployment did not become healthy.",
        hint=f"Last health status: {last}. Check agentbricks deployments logs {name}.",
    )
