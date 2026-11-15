"""``databricks-anthropic``: serve Claude Managed Agents sessions from your laptop or a VM, as your Databricks identity.

    databricks-anthropic init --profile dev --environment env_... [--mcp genie-one] [--with-api-key]
    databricks-anthropic doctor
    databricks-anthropic run [--drain] [--force]
    databricks-anthropic list
    databricks-anthropic cleanup [--dry-run] [--sessions-gone] [--legacy-prefix cma-]

Aliases: ``sandbox-dispatcher`` (run), ``sandbox-list`` (list), ``sandbox-cleanup`` (cleanup).

``init`` stores settings in ``~/.config/databricks-anthropic/<name>.toml`` and the keys in ``<name>.secrets``
(mode 0600). Keys are prompted for with a hidden prompt, or taken from ANTHROPIC_ENVIRONMENT_KEY and
ANTHROPIC_API_KEY. Flags override the file for one run. Every session on the environment runs with the
identity of ``--profile``, so that profile is always named explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from databricks_anthropic.cli import auth, config, doctor, lock
from databricks_anthropic.cli.config import (
    DEFAULT_NAME,
    LIFECYCLES,
    MCP_SERVERS,
    ConfigError,
    Secrets,
    Settings,
)
from databricks_anthropic.sandbox.dispatcher import DatabricksSandboxDispatcher
from databricks_anthropic.sandbox.launcher import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DatabricksSandboxLauncher,
    Lifecycle,
)

log = logging.getLogger("databricks_anthropic.cli")

DESCRIPTION = """\
Serve Claude Managed Agents sessions from this machine, as your Databricks identity.

  databricks-anthropic init --profile dev --environment env_... [--mcp genie-one] [--with-api-key]
  databricks-anthropic doctor
  databricks-anthropic run [--drain] [--force]
  databricks-anthropic list
  databricks-anthropic cleanup [--dry-run] [--sessions-gone] [--legacy-prefix cma-]

Aliases: sandbox-dispatcher (run), sandbox-list (list), sandbox-cleanup (cleanup).
Settings: ~/.config/databricks-anthropic/<name>.toml. Keys: <name>.secrets (mode 0600).
Every session on the environment runs with the identity of --profile: that profile is always named explicitly.
"""

ALIASES = {"sandbox-dispatcher": "run", "sandbox-list": "list", "sandbox-cleanup": "cleanup"}
ENVIRONMENT_KEY_VAR = "ANTHROPIC_ENVIRONMENT_KEY"
API_KEY_VAR = "ANTHROPIC_API_KEY"
# A worker that just stopped still counts as polling for about 30 s after its last poll.
POLL_WINDOW_S = 35
DEFAULT_INACTIVITY_TIMEOUT_S = 600
DEFAULT_MAX_IDLE_S = 300.0


def _pause(seconds: float) -> None:
    """The double-serving wait; a module-level indirection so tests can replace it."""
    time.sleep(seconds)


class ConcurrentWorkerError(RuntimeError):
    def __init__(self, environment_id: str, count: int) -> None:
        self.environment_id = environment_id
        self.count = count
        super().__init__(
            f"refusing to start: {count} other worker(s) are polling environment {environment_id}, so two "
            "dispatchers would share its queue. Stop the other dispatcher, or pass --force if you know it is stale."
        )


def check_no_other_worker(
    read_polling: Callable[[], int],
    *,
    environment_id: str,
    force: bool,
    sleep: Callable[[float], None] = _pause,
    wait_s: float = POLL_WINDOW_S,
) -> None:
    """Refuse to serve an environment another worker is already polling.

    ``read_polling`` is the queue's ``workers_polling``. A worker that just stopped still counts for about 30 s, so
    a non-zero count is re-read after ``wait_s``. A failed read warns and carries on: the check is a safeguard,
    not a gate that can lock a user out.
    """
    try:
        first = read_polling()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "could not read the queue stats (%s: %s); starting without the double-serving check",
            type(e).__name__,
            e,
        )
        return
    if first == 0:
        return
    log.warning(
        "%d worker(s) already polling environment %s; waiting %ds in case one has just stopped",
        first,
        environment_id,
        int(wait_s),
    )
    sleep(wait_s)
    try:
        second = read_polling()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "could not re-read the queue stats (%s: %s); starting anyway", type(e).__name__, e
        )
        return
    if second == 0:
        log.info("the other worker has stopped")
        return
    if force:
        log.warning(
            "--force: starting with %d other worker(s) still polling environment %s",
            second,
            environment_id,
        )
        return
    raise ConcurrentWorkerError(environment_id, second)


async def serve(dispatcher: Any, *, force_exit: Callable[[int], None] = os._exit) -> int:
    """Run the dispatcher until it ends or a signal stops it.

    The first SIGINT or SIGTERM cancels the dispatcher task. Its ``finally`` then waits for the launches it has
    already claimed, so no session turn is dropped. A second signal exits at once.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(dispatcher.run())
    received = 0

    def on_signal(name: str) -> None:
        nonlocal received
        received += 1
        if received > 1:
            log.error("second %s: exiting now without waiting for claimed launches", name)
            sys.stderr.flush()
            force_exit(130)
            return
        log.info("%s: finishing claimed launches, then stopping (send it again to exit now)", name)
        task.cancel()

    signals = (signal.SIGINT, signal.SIGTERM)
    for sig in signals:
        loop.add_signal_handler(sig, on_signal, sig.name)
    try:
        await task
    except asyncio.CancelledError:
        log.info("stopped")
    finally:
        for sig in signals:
            loop.remove_signal_handler(sig)
    return 1 if dispatcher.failures else 0


# ------------------------------------------------------------------ configuration in use


@dataclass
class Context:
    name: str
    directory: Path
    settings: Settings
    secrets: Secrets


def _config_name(args: argparse.Namespace) -> str:
    try:
        return config.check_name(args.config)
    except ConfigError as e:
        sys.exit(str(e))


def _load(
    args: argparse.Namespace, *, required: bool = True, check_secrets_mode: bool = True
) -> Context:
    """The saved configuration with this invocation's flags applied on top."""
    name = _config_name(args)
    directory = config.config_dir(args.config_dir)
    toml_path, secrets_path = config.paths(directory, name)
    settings = Settings()
    if toml_path.exists():
        try:
            settings = config.load_settings(toml_path)
        except ConfigError as e:
            sys.exit(str(e))
    elif required:
        sys.exit(
            f"no configuration at {toml_path}. Run: databricks-anthropic init --profile <name> --environment env_..."
        )
    try:
        settings = config.apply_overrides(
            settings,
            profile=args.profile,
            mcp_servers=getattr(args, "mcp", None),
            lifecycle=getattr(args, "lifecycle", None),
            retention_days=getattr(args, "retention_days", None),
            export_volume=getattr(args, "export_volume", None),
            anthropic_base_url=args.anthropic_base_url,
        )
        secrets = (
            config.load_secrets(secrets_path, check_mode=check_secrets_mode)
            if secrets_path.exists()
            else Secrets()
        )
    except ConfigError as e:
        sys.exit(str(e))
    return Context(name=name, directory=directory, settings=settings, secrets=secrets)


def _environment_id(ctx: Context) -> str:
    env_id = ctx.settings.environment_id
    if not env_id:
        raise SystemExit(
            "no environment id saved: run databricks-anthropic init --environment env_..."
        )
    return env_id


def _environment_key(ctx: Context) -> str:
    key = ctx.secrets.environment_key
    if not key:
        raise SystemExit(
            "no environment key saved: run databricks-anthropic init --environment env_... again"
        )
    return key


def _login_kwargs(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    return {"profile": ctx.settings.profile}


def _login(ctx: Context, args: argparse.Namespace) -> Any:
    try:
        return auth.login(**_login_kwargs(ctx, args))
    except auth.AuthSetupError as e:
        sys.exit(f"Databricks login is not set up.\n\n{e}")


def _mcp_servers(w: Any, names: list[str]) -> list[Any]:
    if not names:
        return []
    try:
        from databricks_anthropic.sandbox.mcp import genie_one_server
    except ImportError:
        sys.exit(f"--mcp needs the MCP extra: {doctor.MCP_HINT}")
    if doctor.mcp_extra_missing():
        sys.exit(f"--mcp needs the MCP extra: {doctor.MCP_HINT}")
    # MCP_SERVERS has one entry today; each name maps to its server here.
    return [genie_one_server(w) for name in names if name == "genie-one"]


def _launcher(
    ctx: Context,
    w: Any,
    args: argparse.Namespace,
    *,
    environment_key: str,
    mcp_servers: list[Any],
    legacy_prefix: str | None = None,
) -> DatabricksSandboxLauncher:
    s = ctx.settings
    try:
        return DatabricksSandboxLauncher(
            workspace_client=w,
            environment_id=s.environment_id or "",
            environment_key=environment_key,
            anthropic_base_url=s.anthropic_base_url,
            inactivity_timeout_s=getattr(args, "inactivity_timeout", None)
            or DEFAULT_INACTIVITY_TIMEOUT_S,
            max_idle_s=getattr(args, "max_idle", None) or DEFAULT_MAX_IDLE_S,
            image_uri=getattr(args, "image_uri", None),
            index_url=getattr(args, "index_url", None),
            mcp_servers=mcp_servers,
            lifecycle=cast(Lifecycle, s.lifecycle),
            retention_days=s.retention_days,
            export_volume_path=s.export_volume,
            # The HMAC key names sandboxes; a legacy prefix (the old unsalted `cma-` names) has no key.
            sandbox_id_prefix=legacy_prefix or s.sandbox_prefix,
            sandbox_id_key=None if legacy_prefix else ctx.secrets.sandbox_id_key,
        )
    except ValueError as e:
        sys.exit(str(e))


def _sessions_api(ctx: Context) -> Any:
    if not ctx.secrets.api_key:
        sys.exit("--sessions-gone needs an API key: run databricks-anthropic init --with-api-key")
    import anthropic

    return anthropic.Anthropic(
        api_key=ctx.secrets.api_key, base_url=ctx.settings.anthropic_base_url
    )


# ------------------------------------------------------------------ commands


def _secret_from(env_var: str, label: str, keep: str | None) -> str | None:
    """A key from the environment variable if set, else a hidden prompt. Never echoed."""
    value = (os.environ.get(env_var) or "").strip()
    if value:
        return value
    hint = " (Enter to keep the saved one)" if keep else " (input hidden)"
    return getpass.getpass(f"{label}{hint}: ").strip() or None


def cmd_init(args: argparse.Namespace) -> int:
    name = _config_name(args)
    directory = config.ensure_config_dir(config.config_dir(args.config_dir))
    toml_path, secrets_path = config.paths(directory, name)
    try:
        existing = config.load_settings(toml_path) if toml_path.exists() else None
        # Read without the mode check: init rewrites the file 0600, which also repairs a too-open one.
        old = (
            config.load_secrets(secrets_path, check_mode=False)
            if secrets_path.exists()
            else Secrets()
        )
        settings = config.apply_overrides(
            existing or Settings(),
            profile=args.profile,
            environment_id=args.environment,
            mcp_servers=args.mcp,
            lifecycle=args.lifecycle,
            retention_days=args.retention_days,
            export_volume=args.export_volume,
            anthropic_base_url=args.anthropic_base_url,
        )
    except ConfigError as e:
        sys.exit(str(e))
    if not settings.profile:
        sys.exit(
            "init needs --profile <name>, a profile created by `databricks auth login`.\n\n"
            + auth.no_profile_instructions()
        )
    if not settings.environment_id:
        sys.exit(
            "init needs --environment env_... (the environment id from Claude Console -> Environments)"
        )

    environment_key = _secret_from(
        ENVIRONMENT_KEY_VAR, "Anthropic environment key", old.environment_key
    )
    if environment_key is None:
        if not old.environment_key:
            sys.exit(
                f"no environment key given: enter it at the prompt, or set {ENVIRONMENT_KEY_VAR}"
            )
        environment_key = old.environment_key
    api_key = old.api_key
    if args.with_api_key:
        api_key = _secret_from(API_KEY_VAR, "Anthropic API key (optional)", old.api_key) or api_key
    # Generated once: a new key would change every sandbox name and orphan the resumable ones.
    sandbox_id_key = old.sandbox_id_key
    if not sandbox_id_key:
        sandbox_id_key = config.new_sandbox_id_key()
        if existing is not None:
            print(
                "note: no sandbox-name key was saved for this config, so a new one was generated. "
                "Sandboxes created before this are not matched by cleanup or resume."
            )
    try:
        config.save_secrets(
            secrets_path,
            Secrets(
                environment_key=environment_key, sandbox_id_key=sandbox_id_key, api_key=api_key
            ),
        )
        config.save_settings(toml_path, settings)
    except (ConfigError, OSError) as e:
        sys.exit(f"could not save the configuration: {e}")

    print(f"Saved {toml_path} and {secrets_path} (0600). Keys are not shown.")
    print(
        f"  profile: {settings.profile or '-'}   environment: {settings.environment_id}   "
        f"mcp: {', '.join(settings.mcp_servers) or '-'}"
    )
    print("Next: databricks-anthropic doctor")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ctx = _load(args, required=False, check_secrets_mode=False)
    return doctor.run_doctor(
        config_dir=ctx.directory,
        name=ctx.name,
        settings=ctx.settings,
        secrets=ctx.secrets,
        login_kwargs=_login_kwargs(ctx, args),
        mcp_names=ctx.settings.mcp_servers,
    )


def cmd_run(args: argparse.Namespace) -> int:
    ctx = _load(args)
    env_id = _environment_id(ctx)
    environment_key = _environment_key(ctx)
    if args.sessions_gone and not ctx.secrets.api_key:
        sys.exit("--sessions-gone needs an API key: run databricks-anthropic init --with-api-key")

    try:
        instance = lock.EnvironmentLock(lock.lock_path(ctx.directory, env_id), env_id)
        instance.acquire()
    except lock.LockUnsupported as e:
        sys.exit(str(e))
    except lock.LockHeld as e:
        sys.exit(
            str(e) + "\nStop it first (or run databricks-anthropic doctor to see who holds it)."
        )
    try:
        w = _login(ctx, args)
        servers = _mcp_servers(w, ctx.settings.mcp_servers)
        api = doctor.environment_api(environment_key, ctx.settings.anthropic_base_url)
        try:
            check_no_other_worker(
                lambda: doctor.workers_polling(api, env_id),
                environment_id=env_id,
                force=args.force,
                sleep=_pause,
            )
        except ConcurrentWorkerError as e:
            sys.exit(str(e))
        user = w.current_user.me().user_name
        print(
            f"serving environment {env_id} as {user}: every session on this environment runs with this "
            "identity's Databricks access",
            flush=True,
        )
        launcher = _launcher(ctx, w, args, environment_key=environment_key, mcp_servers=servers)
        dispatcher = DatabricksSandboxDispatcher(
            launcher,
            max_concurrent_launches=args.max_concurrent_launches,
            drain=args.drain,
            sweep_interval_s=args.sweep_interval or None,
            sessions_api=_sessions_api(ctx) if args.sessions_gone else None,
        )
        try:
            return asyncio.run(serve(dispatcher))
        except Exception as e:  # noqa: BLE001 - one line for the user; the traceback stays in -v output
            log.error(
                "dispatcher stopped: %s: %s", type(e).__name__, e, exc_info=bool(args.verbose)
            )
            return 1
    finally:
        instance.release()


def cmd_list(args: argparse.Namespace) -> int:
    ctx = _load(args)
    _environment_id(ctx)
    w = _login(ctx, args)
    dispatcher = DatabricksSandboxDispatcher(
        _launcher(ctx, w, args, environment_key="", mcp_servers=[])
    )
    for sb in dispatcher.list_session_sandboxes():
        tags = dict(getattr(getattr(sb, "metadata", None), "custom_tags", None) or {})
        state = sb.status.state.value if sb.status and sb.status.state else "?"
        print(
            f"{sb.name}\t{state}\tsession={tags.get('claude_session_id', '?')}\t"
            f"lifecycle={tags.get('lifecycle', '?')}"
        )
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    ctx = _load(args)
    _environment_id(ctx)
    if args.delete_all and not (args.yes or args.dry_run):
        sys.exit(
            "--delete-all removes every session sandbox and its files; add --yes (or --dry-run)"
        )
    w = _login(ctx, args)
    if args.legacy_prefix:
        print(
            f"scanning sandboxes named {args.legacy_prefix}* (legacy names, no key); this environment only"
        )
    launcher = _launcher(
        ctx, w, args, environment_key="", mcp_servers=[], legacy_prefix=args.legacy_prefix
    )
    dispatcher = DatabricksSandboxDispatcher(launcher)
    plan = dispatcher.cleanup(
        dry_run=args.dry_run,
        delete_all=args.delete_all,
        force=args.force,
        sessions_api=_sessions_api(ctx) if args.sessions_gone else None,
    )
    for action in plan:
        verb = "would delete" if args.dry_run and action.action == "delete" else action.action
        print(f"{verb:12} {action.sandbox_name}  ({action.reason})")
    return 0


# ------------------------------------------------------------------ argument parsing


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=DEFAULT_NAME,
        metavar="NAME",
        help=f"configuration name: {DEFAULT_NAME}.toml and {DEFAULT_NAME}.secrets (default: {DEFAULT_NAME})",
    )
    common.add_argument(
        "--config-dir",
        metavar="PATH",
        help="configuration directory (default: $DATABRICKS_ANTHROPIC_CONFIG_DIR or ~/.config/databricks-anthropic)",
    )
    common.add_argument(
        "--profile",
        help="Databricks CLI profile the sessions run as, created by `databricks auth login --host <workspace-url>`",
    )
    common.add_argument(
        "--anthropic-base-url",
        default=None,
        help=f"Anthropic API URL (default: {DEFAULT_ANTHROPIC_BASE_URL}; an ambient ANTHROPIC_BASE_URL is ignored)",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return common


def _mcp_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("--mcp", action="append", choices=list(MCP_SERVERS), help=help_text)


def _lifecycle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lifecycle",
        choices=list(LIFECYCLES),
        help="ephemeral (delete the sandbox when idle) or resumable (stop it and keep its files)",
    )
    parser.add_argument(
        "--retention-days", type=float, help="resumable: delete stopped sandboxes unused this long"
    )
    parser.add_argument(
        "--export-volume",
        help="UC Volume folder for workdir/outputs, e.g. /Volumes/main/agents/out",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="databricks-anthropic",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    init = sub.add_parser(
        "init", parents=[_common()], help="save the configuration and keys (keys are prompted)"
    )
    init.add_argument(
        "--environment", metavar="ENV_ID", help="self-hosted environment id (env_...)"
    )
    _mcp_arg(init, "serve a Databricks MCP server's tools from the worker (genie-one)")
    _lifecycle_args(init)
    init.add_argument(
        "--with-api-key", action="store_true", help="also store an Anthropic API key (prompted)"
    )
    init.set_defaults(func=cmd_init)

    doc = sub.add_parser(
        "doctor", parents=[_common()], help="check every prerequisite and explain each fix"
    )
    _mcp_arg(doc, "also check MCP server access (genie-one)")
    doc.set_defaults(func=cmd_doctor)

    run = sub.add_parser(
        "run",
        aliases=["sandbox-dispatcher"],
        parents=[_common()],
        help="serve sessions on the environment until Ctrl-C or SIGTERM",
    )
    _mcp_arg(run, "serve a Databricks MCP server's tools from the worker (genie-one)")
    run.add_argument("--drain", action="store_true", help="exit once the queue is empty")
    run.add_argument(
        "--force",
        action="store_true",
        help="start even if another worker is polling this environment",
    )
    run.add_argument("--max-concurrent-launches", type=int, default=8)
    run.add_argument(
        "--max-idle",
        type=float,
        default=DEFAULT_MAX_IDLE_S,
        help="worker idle seconds before it returns",
    )
    run.add_argument(
        "--inactivity-timeout",
        type=int,
        default=DEFAULT_INACTIVITY_TIMEOUT_S,
        help="sandbox auto-stop safety net (s)",
    )
    run.add_argument("--image-uri", help="Artifact Registry image (catalog.schema.artifact:tag)")
    _lifecycle_args(run)
    run.add_argument(
        "--sweep-interval",
        type=float,
        default=3600,
        help="seconds between cleanup sweeps (0 = off)",
    )
    run.add_argument(
        "--sessions-gone",
        action="store_true",
        help="also delete sandboxes of deleted or archived sessions (needs the saved API key)",
    )
    run.add_argument("--index-url", help="package index for uv inside the sandbox")
    run.set_defaults(func=cmd_run)

    lst = sub.add_parser(
        "list", aliases=["sandbox-list"], parents=[_common()], help="list session sandboxes"
    )
    lst.set_defaults(func=cmd_list)

    cleanup = sub.add_parser(
        "cleanup",
        aliases=["sandbox-cleanup"],
        parents=[_common()],
        help="delete session sandboxes per the lifecycle",
    )
    cleanup.add_argument("--dry-run", action="store_true", help="show what would be deleted")
    cleanup.add_argument(
        "--retention-days", type=float, help="delete stopped sandboxes unused this long"
    )
    cleanup.add_argument(
        "--sessions-gone",
        action="store_true",
        help="delete sandboxes whose Claude session was deleted or archived (needs the saved API key)",
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
    cleanup.add_argument(
        "--legacy-prefix",
        metavar="PREFIX",
        help="plan cleanup for sandboxes named PREFIX<unsalted hash> from the old dev CLI (e.g. cma-); "
        "use --dry-run first",
    )
    cleanup.set_defaults(func=cmd_cleanup)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.command = ALIASES.get(
        args.command, args.command
    )  # an alias parses exactly like its command
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not args.verbose:
        for noisy in ("httpx", "httpx2", "databricks.sdk"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    handler: Callable[[argparse.Namespace], int] = args.func
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
