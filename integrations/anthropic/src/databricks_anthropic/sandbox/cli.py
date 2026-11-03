"""``databricks-anthropic`` command line.

    databricks-anthropic doctor --profile my-workspace [--mcp genie-one]
    databricks-anthropic sandbox-dispatcher --profile my-workspace [--mcp genie-one] [--drain]
    databricks-anthropic sandbox-dispatcher --profile my-workspace --lifecycle resumable --retention-days 7
    databricks-anthropic sandbox-list --profile my-workspace
    databricks-anthropic sandbox-cleanup --profile my-workspace [--dry-run] [--sessions-gone] [--retention-days N]

Anthropic values come from ``--anthropic-env-file`` (dotenv) or the process environment:
``ANTHROPIC_ENVIRONMENT_ID`` and ``ANTHROPIC_ENVIRONMENT_KEY``. The Anthropic API URL is
``--anthropic-base-url`` (default https://api.anthropic.com); an ambient
``ANTHROPIC_BASE_URL`` is deliberately ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from databricks.sdk import WorkspaceClient

from databricks_anthropic import _useragent
from databricks_anthropic import auth as dbx_auth
from databricks_anthropic.envfile import read_env_file
from databricks_anthropic.sandbox.dispatcher import DatabricksSandboxDispatcher
from databricks_anthropic.sandbox.launcher import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DatabricksSandboxLauncher,
)


def _apply_oauth_env(args: argparse.Namespace) -> None:
    """Load DATABRICKS_U2M_CLIENT_ID/SECRET from --oauth-env-file into this process."""
    values = read_env_file(args.oauth_env_file)
    for key in (dbx_auth.OAUTH_CLIENT_ID_ENV, dbx_auth.OAUTH_CLIENT_SECRET_ENV):
        if values.get(key) and not os.environ.get(key):
            os.environ[key] = values[key]


def _workspace_client(args: argparse.Namespace) -> WorkspaceClient:
    """U2M login via CLI profile or SDK browser flow; exits with fix-it steps when not set up."""
    _apply_oauth_env(args)
    secret = os.environ.get(args.oauth_client_secret_env) if args.oauth_client_secret_env else None
    try:
        return dbx_auth.login(
            auth=args.auth,
            profile=args.profile,
            host=args.host,
            client_id=args.oauth_client_id,
            client_secret=secret,
        )
    except dbx_auth.AuthSetupError as e:
        sys.exit(f"Databricks login is not set up.\n\n{e}")


def _launcher(args: argparse.Namespace, *, need_keys: bool = True) -> DatabricksSandboxLauncher:
    env = {**os.environ, **read_env_file(args.anthropic_env_file)}
    environment_id = env.get("ANTHROPIC_ENVIRONMENT_ID", "")
    environment_key = env.get("ANTHROPIC_ENVIRONMENT_KEY", "")
    if not environment_id:
        # Also needed to list or clean up: only sandboxes tagged with this environment are touched.
        sys.exit("ANTHROPIC_ENVIRONMENT_ID is required (environment or --anthropic-env-file)")
    if need_keys and not environment_key:
        sys.exit("ANTHROPIC_ENVIRONMENT_KEY is required (environment or --anthropic-env-file)")
    w = _workspace_client(args)
    mcp_servers = []
    if getattr(args, "mcp", None):
        from databricks_anthropic.sandbox.mcp import genie_one_server

        for name in args.mcp:
            if name != "genie-one":
                sys.exit(f"unknown --mcp {name!r} (supported: genie-one)")
            mcp_servers.append(genie_one_server(w))
    return DatabricksSandboxLauncher(
        workspace_client=w,
        environment_id=environment_id,
        environment_key=environment_key,
        anthropic_base_url=args.anthropic_base_url,
        inactivity_timeout_s=getattr(args, "inactivity_timeout", 600),
        max_idle_s=getattr(args, "max_idle", 300),
        image_uri=getattr(args, "image_uri", None),
        index_url=getattr(args, "index_url", None),
        mcp_servers=mcp_servers,
        lifecycle=getattr(args, "lifecycle", "ephemeral"),
        retention_days=7.0
        if getattr(args, "retention_days", None) is None
        else args.retention_days,
        export_volume_path=getattr(args, "export_volume", None),
    )


def _sessions_api(args: argparse.Namespace):
    """anthropic.Anthropic with the API key (only needed for session-gone cleanup)."""
    import anthropic

    key = {**os.environ, **read_env_file(args.anthropic_env_file)}.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("--sessions-gone needs ANTHROPIC_API_KEY (environment or --anthropic-env-file)")
    return anthropic.Anthropic(api_key=key, base_url=args.anthropic_base_url)


def _doctor(args: argparse.Namespace) -> int:
    """Check Databricks login, Sandbox preview, MCP access and Anthropic settings; print fixes."""
    failed = False

    def report(ok: bool, check: str, detail: str = "") -> None:
        nonlocal failed
        failed |= not ok
        print(f"[{'OK' if ok else 'FIX'}] {check}" + (f"\n      {detail}" if detail else ""))

    path = dbx_auth.cli_path()
    version = dbx_auth.cli_version(path) if path else None
    print(
        f"[INFO] Databricks CLI: {path or 'not installed'}"
        + (f" (v{'.'.join(map(str, version))})" if version else "")
    )
    try:
        _apply_oauth_env(args)
        secret = (
            os.environ.get(args.oauth_client_secret_env) if args.oauth_client_secret_env else None
        )
        w = dbx_auth.login(
            auth=args.auth,
            profile=args.profile,
            host=args.host,
            client_id=args.oauth_client_id,
            client_secret=secret,
        )
        report(
            True,
            "Databricks login",
            f"{w.current_user.me().user_name} on {w.config.host} (auth_type={w.config.auth_type}); "
            "sandboxes will act as this identity",
        )
    except dbx_auth.AuthSetupError as e:
        report(False, "Databricks login", str(e).replace("\n", "\n      "))
        return 1
    try:
        next(iter(w.sandbox.list_sandboxes(page_size=1)), None)  # one page is enough
        report(True, "Databricks Sandbox API")
    except Exception as e:  # noqa: BLE001
        report(
            False,
            "Databricks Sandbox API",
            f"{type(e).__name__}: {e}. A workspace admin must enable the "
            "'Databricks Sandboxes' preview (Settings -> Previews); it can take ~1 minute to apply.",
        )
    if args.mcp:
        from databricks_anthropic.sandbox.mcp import alist_mcp_custom_tools, genie_one_server

        server = genie_one_server(w)
        try:
            names = [t["name"] for t in asyncio.run(alist_mcp_custom_tools(w, server))]
            report(True, "Genie One MCP", f"tools: {names}")
        except Exception as e:  # noqa: BLE001 - includes anyio ExceptionGroups
            report(
                False,
                "Genie One MCP",
                f"{type(e).__name__}: {e}. Needs Genie One enabled and EXECUTE on "
                "system.ai.genie_one_mcp for this identity.",
            )
    env = {**os.environ, **read_env_file(args.anthropic_env_file)}
    have_env = bool(env.get("ANTHROPIC_ENVIRONMENT_ID") and env.get("ANTHROPIC_ENVIRONMENT_KEY"))
    report(
        have_env,
        "Anthropic environment ID + key",
        ""
        if have_env
        else "Set ANTHROPIC_ENVIRONMENT_ID and ANTHROPIC_ENVIRONMENT_KEY (Claude Console -> "
        "Environments -> Self-hosted -> Generate environment key), or pass --anthropic-env-file.",
    )
    if have_env and env.get("ANTHROPIC_API_KEY"):
        import anthropic

        try:
            environment = anthropic.Anthropic(
                api_key=env["ANTHROPIC_API_KEY"], base_url=args.anthropic_base_url
            ).beta.environments.retrieve(env["ANTHROPIC_ENVIRONMENT_ID"])
            hosting = getattr(environment.config, "type", None)
            report(
                hosting == "self_hosted",
                "Anthropic environment is self-hosted",
                f"{environment.name} ({hosting})"
                if hosting == "self_hosted"
                else f"{environment.name} is {hosting}; create one with Hosting = Self-hosted",
            )
        except Exception as e:  # noqa: BLE001
            report(False, "Anthropic environment", f"{type(e).__name__}: {e}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    _useragent.apply_process_env()
    parser = argparse.ArgumentParser(
        prog="databricks-anthropic",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", help="Databricks CLI profile (sandboxes act as this identity)")
    common.add_argument(
        "--auth",
        choices=["cli", "browser", "auto"],
        default="cli",
        help="U2M login: Databricks CLI profile (default), SDK browser login, or auto-detect",
    )
    common.add_argument(
        "--host", help="workspace URL (needed for --auth browser without a profile)"
    )
    common.add_argument(
        "--oauth-client-id",
        help=f"OAuth app for browser login (or ${dbx_auth.OAUTH_CLIENT_ID_ENV}); "
        f"its redirect URL must be {dbx_auth.SDK_BROWSER_REDIRECT_URL}",
    )
    common.add_argument(
        "--oauth-client-secret-env",
        default=dbx_auth.OAUTH_CLIENT_SECRET_ENV,
        help="env var holding the OAuth app secret (confidential apps)",
    )
    common.add_argument("--oauth-env-file", help="dotenv with DATABRICKS_U2M_CLIENT_ID/SECRET")
    common.add_argument(
        "--anthropic-env-file", help="dotenv file with ANTHROPIC_ENVIRONMENT_ID/KEY"
    )
    common.add_argument("--anthropic-base-url", default=DEFAULT_ANTHROPIC_BASE_URL)
    common.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "sandbox-dispatcher", parents=[common], help="poll the queue and launch sandboxes"
    )
    run.add_argument(
        "--mcp",
        action="append",
        choices=["genie-one"],
        help="serve a Databricks MCP server's tools from the worker",
    )
    run.add_argument("--drain", action="store_true", help="exit once the queue is empty")
    run.add_argument("--max-concurrent-launches", type=int, default=8)
    run.add_argument(
        "--max-idle", type=float, default=300, help="worker idle seconds before it returns"
    )
    run.add_argument(
        "--inactivity-timeout", type=int, default=600, help="sandbox auto-stop safety net (s)"
    )
    run.add_argument("--image-uri", help="Artifact Registry image (catalog.schema.artifact:tag)")
    run.add_argument(
        "--lifecycle",
        choices=["ephemeral", "resumable"],
        default="ephemeral",
        help="ephemeral (default): delete the sandbox when the session goes idle; "
        "resumable: stop it (files kept) and resume on the next message",
    )
    run.add_argument(
        "--retention-days",
        type=float,
        default=7.0,
        help="resumable: delete stopped sandboxes unused this long (default 7)",
    )
    run.add_argument(
        "--export-volume",
        help="UC Volume folder; workdir/outputs is copied to <folder>/<session>/ "
        "when the worker finishes (e.g. /Volumes/main/agents/outputs)",
    )
    run.add_argument(
        "--sweep-interval",
        type=float,
        default=3600,
        help="seconds between cleanup sweeps (0 = off)",
    )
    run.add_argument(
        "--sessions-gone",
        action="store_true",
        help="also delete sandboxes of deleted/archived sessions (needs ANTHROPIC_API_KEY)",
    )
    run.add_argument("--index-url", help="package index for uv inside the sandbox")

    doctor = sub.add_parser(
        "doctor", parents=[common], help="check every prerequisite and explain fixes"
    )
    doctor.add_argument(
        "--mcp", action="append", choices=["genie-one"], help="also check MCP server access"
    )

    sub.add_parser("sandbox-list", parents=[common], help="list session sandboxes")
    cleanup = sub.add_parser(
        "sandbox-cleanup", parents=[common], help="delete session sandboxes per the lifecycle"
    )
    cleanup.add_argument("--dry-run", action="store_true", help="show what would be deleted")
    cleanup.add_argument(
        "--retention-days",
        type=float,
        default=7.0,
        help="delete stopped sandboxes unused this long (default 7)",
    )
    cleanup.add_argument(
        "--sessions-gone",
        action="store_true",
        help="delete sandboxes whose Claude session was deleted or archived (needs ANTHROPIC_API_KEY)",
    )
    cleanup.add_argument(
        "--delete-all",
        action="store_true",
        help="delete every stopped sandbox this environment owns (removes their data); needs --yes",
    )
    cleanup.add_argument(
        "--force", action="store_true", help="with --delete-all: also delete running sandboxes"
    )
    cleanup.add_argument("--yes", action="store_true", help="confirm --delete-all")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not args.verbose:
        for noisy in ("httpx", "httpx2", "databricks.sdk"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "doctor":
        return _doctor(args)

    if args.command == "sandbox-dispatcher":
        dispatcher = DatabricksSandboxDispatcher(
            _launcher(args),
            max_concurrent_launches=args.max_concurrent_launches,
            drain=args.drain,
            sweep_interval_s=args.sweep_interval or None,
            sessions_api=_sessions_api(args) if args.sessions_gone else None,
        )
        try:
            asyncio.run(dispatcher.run())
        except KeyboardInterrupt:
            pass
        return 1 if dispatcher.failures else 0

    dispatcher = DatabricksSandboxDispatcher(_launcher(args, need_keys=False))
    if args.command == "sandbox-list":
        for sb in dispatcher.list_session_sandboxes():
            tags = dict(getattr(getattr(sb, "metadata", None), "custom_tags", None) or {})
            state = sb.status.state.value if sb.status and sb.status.state else "?"
            print(
                f"{sb.name}\t{state}\tsession={tags.get('claude_session_id', '?')}\t"
                f"lifecycle={tags.get('lifecycle', '?')}"
            )
        return 0
    if args.delete_all and not (args.yes or args.dry_run):
        sys.exit(
            "--delete-all removes every session sandbox and its files; add --yes (or --dry-run)"
        )
    plan = dispatcher.cleanup(
        dry_run=args.dry_run,
        delete_all=args.delete_all,
        force=args.force,
        retention_days=args.retention_days,
        sessions_api=_sessions_api(args) if args.sessions_gone else None,
    )
    for action in plan:
        verb = "would delete" if args.dry_run and action.action == "delete" else action.action
        print(f"{verb:12} {action.sandbox_name}  ({action.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
