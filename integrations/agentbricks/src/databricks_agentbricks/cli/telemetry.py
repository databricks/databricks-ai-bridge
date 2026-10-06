"""Best effort Agent Bricks CLI invocation telemetry.

The CLI sends one small ``FrontendLogEntry`` payload after each leaf command.  This module keeps
the logging schema and transport isolated from command behavior: telemetry is opt-out, never
prints a diagnostic, and never changes the command's result when authentication or transport is
unavailable.
"""

from __future__ import annotations

import configparser
import json
import os
import platform
import threading
import time
import uuid
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from pathlib import Path
from typing import Any

_DISABLE_ENV = "AGENTBRICKS_DISABLE_TELEMETRY"
_TELEMETRY_PATH = "/telemetry-ext"

# Keep the foreground wait short.  The worker is daemonized so an unavailable endpoint cannot hold
# the CLI process open after this bound expires.
_MAX_FOREGROUND_WAIT_S = 0.25

_FRAMEWORK_ENUMS = {
    "openai": "FRAMEWORK_OPENAI",
    "langgraph": "FRAMEWORK_LANGGRAPH",
}
_SERVER_ENUMS = {
    "agentbricks": "SERVER_AGENTBRICKS",
    "custom": "SERVER_CUSTOM",
}


def telemetry_disabled() -> bool:
    """Whether the user opted out of CLI telemetry.

    Empty, ``0``, ``false``, ``no``, and ``off`` are false.  Other non-empty values are treated as
    true so shell conventions such as ``AGENTBRICKS_DISABLE_TELEMETRY=1`` and ``=yes`` both work.
    """

    value = os.environ.get(_DISABLE_ENV, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _package_version() -> str:
    try:
        return _installed_version("databricks-agentbricks")
    except PackageNotFoundError:
        return "unknown"


def _enum_name(value: object, values: dict[str, str], other: str) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        return other
    return values.get(raw.lower(), other)


def _error_category(exc: BaseException) -> str:
    """Map an exception to the bounded enum in ``agentbricks_cli_log.proto``."""

    # Click's parser exceptions contain the user's option/argument text.  We use only their type,
    # never their message, in the emitted record.
    import click

    if isinstance(exc, click.UsageError):
        return "ERROR_CATEGORY_USAGE"
    if isinstance(exc, (KeyboardInterrupt, click.Abort)):
        return "ERROR_CATEGORY_INTERRUPTED"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "ERROR_CATEGORY_DEPENDENCY"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "ERROR_CATEGORY_NETWORK"
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, NotADirectoryError)):
        return "ERROR_CATEGORY_CONFIGURATION"

    # AgentCliError deliberately exposes only a bounded error_code.  Keep this duck-typed so the
    # telemetry module does not add a dependency edge to the CLI error implementation.
    error_code = getattr(exc, "error_code", None)
    if isinstance(error_code, str):
        code = error_code.upper()
        if code in {"UNAUTHENTICATED", "INVALID_TOKEN", "INVALID_CREDENTIALS", "TOKEN_EXPIRED"}:
            return "ERROR_CATEGORY_AUTHENTICATION"
        if code in {"PERMISSION_DENIED", "FORBIDDEN", "UNAUTHORIZED", "NOT_AUTHORIZED"}:
            return "ERROR_CATEGORY_AUTHORIZATION"
        if code in {
            "CANCELLED",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "ABORTED",
            "TIMEOUT",
            "TIMED_OUT",
            "NETWORK_ERROR",
            "CONNECTION_ERROR",
        }:
            return "ERROR_CATEGORY_NETWORK"
        if code in {"INVALID_ARGUMENT", "FAILED_PRECONDITION", "CONFIGURATION_ERROR"}:
            return "ERROR_CATEGORY_CONFIGURATION"

    if isinstance(exc, PermissionError):
        return "ERROR_CATEGORY_CONFIGURATION"
    return "ERROR_CATEGORY_RUNTIME"


def _exit_code(exc: BaseException) -> int:
    if isinstance(exc, KeyboardInterrupt):
        # Click converts an uncaught KeyboardInterrupt to Abort, whose process exit code is 1.
        return 1
    value = getattr(exc, "exit_code", None)
    if isinstance(value, int):
        return value
    value = getattr(exc, "code", None)
    if isinstance(value, int):
        return value
    return 1


def _project_context(ctx: Any) -> tuple[object | None, object | None, object | None]:
    """Read only the safe, enum-like project values put on ``CliContext`` by callbacks."""

    obj = getattr(ctx, "obj", None)
    return (
        getattr(obj, "telemetry_framework", None),
        getattr(obj, "telemetry_server", None),
        getattr(obj, "telemetry_tracing_configured", None),
    )


def build_log(
    ctx: Any,
    *,
    execution_time_ms: int,
    success: bool,
    exit_code: int,
    error_category: str | None = None,
) -> dict[str, Any]:
    """Build the allowlisted snake_case ``AgentBricksCliLog`` JSON object."""

    framework, server, tracing_configured = _project_context(ctx)
    log: dict[str, Any] = {
        "command_path": str(getattr(ctx, "command_path", "agentbricks")),
        "package_version": _package_version(),
        "operating_system": platform.system().lower(),
        "execution_time_ms": max(0, int(execution_time_ms)),
        "exit_code": int(exit_code),
        "success": bool(success),
    }
    if not success:
        log["error_category"] = error_category or "ERROR_CATEGORY_OTHER"
    framework_name = _enum_name(framework, _FRAMEWORK_ENUMS, "FRAMEWORK_OTHER")
    server_name = _enum_name(server, _SERVER_ENUMS, "SERVER_OTHER")
    if framework_name is not None:
        log["framework"] = framework_name
    if server_name is not None:
        log["server"] = server_name
    if tracing_configured is not None:
        log["tracing_configured"] = bool(tracing_configured)
    return log


def build_payload(log: dict[str, Any], *, upload_time_ms: int | None = None) -> dict[str, Any]:
    """Wrap one proto JSON record in the Databricks ``/telemetry-ext`` request shape."""

    inner = json.dumps(
        {
            "frontend_log_event_id": str(uuid.uuid4()),
            "entry": {"agentbricks_cli_log": log},
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "uploadTime": int(time.time() * 1000) if upload_time_ms is None else int(upload_time_ms),
        "items": [],
        "protoLogs": [inner],
    }


def _root_obj(ctx: Any) -> Any:
    try:
        root = ctx.find_root()
    except (AttributeError, RuntimeError):
        root = ctx
    return getattr(root, "obj", getattr(ctx, "obj", None))


def _existing_workspace_client(obj: Any) -> Any | None:
    """Return a client already constructed for the command, without constructing a new one."""

    private_client = getattr(obj, "_client", None)
    if private_client is None:
        return None
    workspace_client = getattr(private_client, "workspace_client", None)
    if workspace_client is not None:
        return workspace_client
    # Small test doubles and future wrappers may expose api_client directly.
    return private_client if getattr(private_client, "api_client", None) is not None else None


def _has_noninteractive_auth(profile: object | None) -> bool:
    """Conservatively decide if constructing a client can use credentials without a login flow."""

    if os.environ.get("DATABRICKS_HOST", "").strip():
        if os.environ.get("DATABRICKS_TOKEN", "").strip():
            return True
        if (
            os.environ.get("DATABRICKS_CLIENT_ID", "").strip()
            and os.environ.get("DATABRICKS_CLIENT_SECRET", "").strip()
        ):
            return True
        if (
            os.environ.get("DATABRICKS_AZURE_CLIENT_ID", "").strip()
            and os.environ.get("DATABRICKS_AZURE_CLIENT_SECRET", "").strip()
        ):
            return True

    selected_profile = profile if isinstance(profile, str) and profile.strip() else None
    selected_profile = selected_profile or os.environ.get("DATABRICKS_CONFIG_PROFILE", "").strip()
    if not selected_profile:
        return False

    # A profile configured for external-browser auth can launch a browser while the SDK client is
    # constructed.  Read the small INI file first and only let clearly non-interactive providers
    # reach WorkspaceClient; an existing command client is still reused above.
    config_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE", Path.home() / ".databrickscfg"))
    parser = configparser.ConfigParser()
    try:
        if not config_path.is_file():
            return False
        parser.read(config_path)
        if not parser.has_section(selected_profile):
            return False
        section = parser[selected_profile]
    except (OSError, configparser.Error):
        return False

    if section.get("token", "").strip():
        return True
    if section.get("client_id", "").strip() and section.get("client_secret", "").strip():
        return True
    if (
        section.get("azure_client_id", "").strip()
        and section.get("azure_client_secret", "").strip()
    ):
        return True
    if section.get("username", "").strip() and section.get("password", "").strip():
        return True
    return section.get("auth_type", "").strip().lower() in {
        "azure-cli",
        "azure-devops-oidc",
        "databricks-cli",
        "env-oidc",
        "file-oidc",
        "github-oidc",
        "google-credentials",
        "metadata-service",
        "model-serving",
        "oauth-m2m",
        "pat",
        "runtime-native-auth",
        "runtime-oauth",
    }


def _workspace_client(ctx: Any) -> Any | None:
    obj = _root_obj(ctx)
    existing = _existing_workspace_client(obj)
    if existing is not None:
        return existing
    profile = getattr(obj, "profile", None)
    if not _has_noninteractive_auth(profile):
        return None
    try:
        # The shared AgentKit transport preserves the workspace-id host override used by profiles
        # that resolve through a regional host.  It still constructs the regular SDK
        # WorkspaceClient underneath, but keeps this telemetry path consistent with command auth.
        from databricks_agentkit._api_client import _workspace_client

        return _workspace_client(profile or None)
    except BaseException:
        return None


def _post(client: Any, payload: dict[str, Any]) -> None:
    api_client = getattr(client, "api_client", None)
    if api_client is None:
        return
    api_client.do("POST", _TELEMETRY_PATH, body=payload)


def _send(ctx: Any, payload: dict[str, Any]) -> None:
    try:
        client = _workspace_client(ctx)
        if client is not None:
            _post(client, payload)
    except BaseException:
        # Telemetry is deliberately silent.  In particular, do not expose transport/auth messages
        # that could contain workspace details or make a successful CLI command appear to fail.
        return


def emit_command(
    ctx: Any,
    *,
    execution_time_ms: int,
    success: bool,
    exit_code: int,
    error_category: str | None = None,
) -> None:
    """Emit one command record in a daemon worker, bounded by a short foreground join."""

    if telemetry_disabled():
        return
    log = build_log(
        ctx,
        execution_time_ms=execution_time_ms,
        success=success,
        exit_code=exit_code,
        error_category=error_category,
    )
    payload = build_payload(log)
    worker = threading.Thread(
        target=_send, args=(ctx, payload), name="agentbricks-telemetry", daemon=True
    )
    worker.start()
    worker.join(_MAX_FOREGROUND_WAIT_S)


__all__ = [
    "build_log",
    "build_payload",
    "emit_command",
    "telemetry_disabled",
]
