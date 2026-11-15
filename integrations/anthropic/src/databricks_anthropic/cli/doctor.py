"""``databricks-anthropic doctor``: every prerequisite for ``run``, each failing check with its fix.

Lines are ``[OK]``, ``[INFO]``, ``[WARN]`` or ``[FIX]``; only ``[FIX]`` makes the exit code 1. Nothing here prints
a key: the checks report names, paths, modes and counts.
"""

from __future__ import annotations

import asyncio
import importlib.util
import stat
from pathlib import Path
from typing import Any

from databricks_anthropic.cli import auth, lock
from databricks_anthropic.cli.config import Secrets, Settings, paths, secrets_mode_problem
from databricks_anthropic.sandbox.launcher import DEFAULT_ANTHROPIC_BASE_URL

MCP_HINT = "pip install 'databricks-anthropic[mcp]'"


def mcp_extra_missing() -> bool:
    """True when the [mcp] extra (the ``mcp`` package) is not installed."""
    return importlib.util.find_spec("mcp") is None


def environment_api(environment_key: str, base_url: str) -> Any:
    """An Anthropic client that authenticates with the environment key (queue and stats calls)."""
    import anthropic

    # Explicit auth_token and base_url: ambient ANTHROPIC_* (e.g. Claude Code routed through a gateway) must not
    # redirect or re-authenticate queue calls. Same as the dispatcher's own client.
    return anthropic.Anthropic(auth_token=environment_key, base_url=base_url)


def workers_polling(api: Any, environment_id: str) -> int:
    """How many workers are polling the environment's queue right now (a just-stopped one counts for ~30 s)."""
    stats = api.beta.environments.work.stats(environment_id)
    return int(getattr(stats, "workers_polling", None) or 0)


class Report:
    """Prints check lines and remembers whether any of them is a FIX."""

    def __init__(self) -> None:
        self.failed = False

    def _line(self, status: str, check: str, detail: str = "") -> None:
        print(f"[{status}] {check}" + (f"\n      {detail}" if detail else ""))

    def ok(self, check: str, detail: str = "") -> None:
        self._line("OK", check, detail)

    def info(self, check: str, detail: str = "") -> None:
        self._line("INFO", check, detail)

    def warn(self, check: str, detail: str = "") -> None:
        self._line("WARN", check, detail)

    def fix(self, check: str, detail: str = "") -> None:
        self.failed = True
        self._line("FIX", check, detail)


def _private_dir(path: Path) -> bool:
    return not (stat.S_IMODE(path.stat().st_mode) & 0o077)


def run_doctor(
    *,
    config_dir: Path,
    name: str,
    settings: Settings,
    secrets: Secrets,
    login_kwargs: dict[str, Any],
    mcp_names: list[str],
) -> int:
    rep = Report()
    toml_path, secrets_path = paths(config_dir, name)

    if not toml_path.exists():
        rep.fix(
            "Config file",
            f"not found: {toml_path}. Run: databricks-anthropic init --profile <name> --environment env_...",
        )
        return 1
    rep.ok("Config file", str(toml_path))
    if config_dir.exists() and not _private_dir(config_dir):
        rep.warn("Config directory is private (0700)", f"chmod 700 {config_dir}")
    else:
        rep.ok("Config directory is private (0700)")
    if not secrets_path.exists():
        rep.fix("Secrets file", f"not found: {secrets_path}. Run init again.")
    elif (problem := secrets_mode_problem(secrets_path)) is not None:
        rep.fix("Secrets file is private (0600)", problem)
    else:
        rep.ok("Secrets file is private (0600)", str(secrets_path))

    if settings.backend == "mcp":
        rep.fix(
            "Backend",
            'mcp: experimental mcp backend not available yet; set backend = "sdk" (the default)',
        )
    else:
        rep.ok("Backend", "sdk (one Databricks Sandbox per session)")

    # The environment and the local lock need no Databricks login, so they are checked first.
    env_id = settings.environment_id
    have_env = bool(env_id and secrets.environment_key)
    if have_env:
        rep.ok("Anthropic environment ID + key", str(env_id))
        _check_queue(rep, settings, secrets)
        _check_lock(rep, config_dir, env_id)
    else:
        rep.fix(
            "Anthropic environment ID + key",
            "Run init with --environment env_... (the key is prompted, hidden). Get the key from Claude Console -> "
            "Environments -> Self-hosted -> Generate environment key.",
        )

    if not settings.profile:
        rep.fix("Databricks profile", auth.no_profile_instructions().strip())
        return 1

    path = auth.cli_path()
    version = auth.cli_version(path) if path else None
    rep.info(
        "Databricks CLI",
        (path or "not installed") + (f" (v{'.'.join(map(str, version))})" if version else ""),
    )
    try:
        w = auth.login(**login_kwargs)
    except auth.AuthSetupError as e:
        rep.fix("Databricks login", str(e).replace("\n", "\n      "))
        return 1
    rep.ok(
        "Databricks login",
        f"{w.current_user.me().user_name} on {w.config.host} (auth_type={w.config.auth_type}); "
        "sandboxes will act as this identity",
    )

    try:
        next(iter(w.sandbox.list_sandboxes(page_size=1)), None)  # one page is enough
        rep.ok("Databricks Sandbox API")
    except Exception as e:  # noqa: BLE001
        rep.fix(
            "Databricks Sandbox API",
            f"{type(e).__name__}: {e}. A workspace admin must enable the "
            "'Databricks Sandboxes' preview (Settings -> Previews); it can take ~1 minute to apply.",
        )

    if mcp_names:
        if mcp_extra_missing():
            rep.fix("Genie One MCP", f"the MCP extra is not installed: {MCP_HINT}")
        else:
            _check_mcp(rep, w, mcp_names)

    if secrets.api_key and have_env:
        _check_self_hosted(rep, settings, secrets)
    return 1 if rep.failed else 0


def _check_mcp(rep: Report, w: Any, names: list[str]) -> None:
    from databricks_anthropic.sandbox.mcp import alist_mcp_custom_tools, genie_one_server

    server = genie_one_server(w)
    try:
        tools = [t["name"] for t in asyncio.run(alist_mcp_custom_tools(w, server))]
        rep.ok("Genie One MCP", f"tools: {tools}")
    except Exception as e:  # noqa: BLE001 - includes anyio ExceptionGroups
        rep.fix(
            "Genie One MCP",
            f"{type(e).__name__}: {e}. Needs Genie One enabled and EXECUTE on "
            "system.ai.genie_one_mcp for this identity.",
        )


def _check_queue(rep: Report, settings: Settings, secrets: Secrets) -> None:
    base = settings.anthropic_base_url or DEFAULT_ANTHROPIC_BASE_URL
    try:
        n = workers_polling(
            environment_api(secrets.environment_key or "", base), settings.environment_id or ""
        )
    except Exception as e:  # noqa: BLE001
        rep.warn("Work queue", f"could not read the queue stats: {type(e).__name__}: {e}")
        return
    if n == 0:
        rep.ok("Work queue", f"no worker is polling environment {settings.environment_id}")
    else:
        rep.warn(
            "Work queue",
            f"{n} worker(s) polling environment {settings.environment_id}. `run` refuses to start unless "
            "they stop (or it is given --force). Stop the other dispatcher first.",
        )


def _check_lock(rep: Report, config_dir: Path, environment_id: str | None) -> None:
    try:
        pid = lock.holder_pid(config_dir, environment_id or "")
    except lock.LockUnsupported as e:
        rep.warn("Single-instance lock", str(e))
        return
    except OSError as e:
        rep.warn("Single-instance lock", f"could not probe {config_dir}: {e}")
        return
    if pid is None:
        rep.ok(
            "Single-instance lock",
            "free: no dispatcher is running on this machine for this environment",
        )
    else:
        rep.fix(
            "Single-instance lock",
            f"another dispatcher for env {environment_id} is running (pid {pid or 'unknown'}). Stop it first.",
        )


def _check_self_hosted(rep: Report, settings: Settings, secrets: Secrets) -> None:
    import anthropic

    base = settings.anthropic_base_url or DEFAULT_ANTHROPIC_BASE_URL
    try:
        environment = anthropic.Anthropic(
            api_key=secrets.api_key, base_url=base
        ).beta.environments.retrieve(settings.environment_id or "")
    except Exception as e:  # noqa: BLE001
        rep.fix("Anthropic environment", f"{type(e).__name__}: {e}")
        return
    hosting = getattr(environment.config, "type", None)
    if hosting == "self_hosted":
        rep.ok("Anthropic environment is self-hosted", f"{environment.name} ({hosting})")
    else:
        rep.fix(
            "Anthropic environment is self-hosted",
            f"{environment.name} is {hosting}; create one with Hosting = Self-hosted",
        )
