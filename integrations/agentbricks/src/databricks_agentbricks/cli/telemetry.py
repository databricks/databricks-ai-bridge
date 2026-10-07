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

import click

_DISABLE_ENV = "AGENTBRICKS_DISABLE_TELEMETRY"
_TELEMETRY_PATH = "/telemetry-ext"

# Keep the foreground wait short.  The worker is daemonized so an unavailable endpoint cannot hold
# the CLI process open after this bound expires.
_MAX_FOREGROUND_WAIT_S = 0.25

# Parameter values are opt-in. A Click ``Choice`` is not automatically safe to report because a
# command can construct one from a workspace value. Keep this map literal and keyed by the static
# command path produced below so every value sent in telemetry receives an explicit review.
_SAFE_CHOICE_VALUES: dict[str, frozenset[str]] = {
    "agentbricks.output": frozenset({"text", "json"}),
    "agentbricks init.framework": frozenset({"langgraph", "openai"}),
    "agentbricks init.server": frozenset({"agentbricks", "custom"}),
    "agentbricks tools add sandbox.permission": frozenset({"read_only", "read_write"}),
    "agentbricks tools add sandbox.auth": frozenset({"user", "app"}),
    "agentbricks tools add mcp.auth": frozenset({"user", "app"}),
    "agentbricks tools add genie-one.auth": frozenset({"user", "app"}),
    "agentbricks tools add genie-agent.auth": frozenset({"user", "app"}),
    "agentbricks tools list.kind": frozenset(
        {"sandbox", "mcp", "uc-function", "genie-one", "genie-agent"}
    ),
}

# Keep integer telemetry explicitly bounded even when the Click range later changes. This is a
# privacy review boundary, rather than an inferred copy of the command's type declaration.
_SAFE_BOUNDED_INT_RANGES: dict[str, tuple[int, int]] = {
    "agentbricks deploy.instances": (1, 5),
}

# Boolean values are also opt-in. Keep this literal allowlist limited to the reviewed, low-risk
# init flags; future flags contribute only their static name until explicitly reviewed.
_SAFE_BOOL_PARAMETER_NAMES = frozenset(
    {
        "agentbricks init.existing",
        "agentbricks init.disable_chat_app",
    }
)

_MAX_PARAMETER_ENTRIES = 64
_MAX_CONTEXT_DEPTH = 32
_MAX_STATIC_NAME_LENGTH = 128

_ERROR_CATEGORIES = frozenset(
    {
        "ERROR_CATEGORY_OTHER",
        "ERROR_CATEGORY_USAGE",
        "ERROR_CATEGORY_INTERRUPTED",
        "ERROR_CATEGORY_DEPENDENCY",
        "ERROR_CATEGORY_NETWORK",
        "ERROR_CATEGORY_CONFIGURATION",
        "ERROR_CATEGORY_AUTHENTICATION",
        "ERROR_CATEGORY_AUTHORIZATION",
        "ERROR_CATEGORY_RUNTIME",
    }
)


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


def _bounded_error_category(value: object | None) -> str:
    return (
        value if isinstance(value, str) and value in _ERROR_CATEGORIES else "ERROR_CATEGORY_OTHER"
    )


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
    if isinstance(exc, SystemExit) and exc.code is None:
        return 0
    value = getattr(exc, "exit_code", None)
    if isinstance(value, int):
        return value
    value = getattr(exc, "code", None)
    if isinstance(value, int):
        return value
    return 1


def _is_static_name(value: object) -> bool:
    """Whether a Click command name is safe to put in an event."""

    if not isinstance(value, str) or not value or len(value) > _MAX_STATIC_NAME_LENGTH:
        return False
    # Click command names are code-defined.  Still reject whitespace and punctuation that could
    # indicate a value-derived name in a custom command tree.
    return all(character.isalnum() or character in "-_." for character in value)


def _context_chain(ctx: Any) -> tuple[Any, ...]:
    """Return the Click context chain from the root to ``ctx``.

    A malformed test double or third-party Click extension can create a parent cycle.  Treat that
    as an unusable context rather than walking arbitrary objects or hanging telemetry.
    """

    contexts: list[Any] = []
    current = ctx
    for _ in range(_MAX_CONTEXT_DEPTH):
        if current is None:
            return tuple(reversed(contexts))
        if any(current is existing for existing in contexts):
            return ()
        contexts.append(current)
        try:
            current = getattr(current, "parent", None)
        except BaseException:
            return ()
    return ()


def _static_command_path(contexts: tuple[Any, ...]) -> str | None:
    """Build a command path from Click command names, never from argv or parsed arguments."""

    names: list[str] = []
    for context in contexts:
        try:
            command = getattr(context, "command", None)
            name = getattr(command, "name", None)
        except BaseException:
            return None
        if not isinstance(name, str) or not _is_static_name(name):
            return None
        names.append(name)
    if not names:
        return None
    path = " ".join(names)
    return path if len(path) <= _MAX_STATIC_NAME_LENGTH else None


def _parameter_source(context: Any, name: str) -> object | None:
    """Read Click's source marker without falling back to argv or ``ctx.args``."""

    try:
        get_parameter_source = context.get_parameter_source
        return get_parameter_source(name) if callable(get_parameter_source) else None
    except BaseException:
        return None


def _supplied_on_command_line(source: object | None) -> bool:
    # Comparing the enum name keeps this compatible with Click's supported versions while treating
    # a malformed source marker as untrusted.
    return getattr(source, "name", None) == "COMMANDLINE"


def _parameter_value(context: Any, name: str) -> tuple[bool, object]:
    try:
        params = context.params
        if not isinstance(params, dict) or name not in params:
            return False, None
        return True, params[name]
    except BaseException:
        return False, None


def _parameter_entry(*, name: str, parameter: Any, value: object) -> dict[str, Any] | None:
    """Create one bounded parameter entry, dropping every unsafe value."""

    try:
        if bool(getattr(parameter, "hidden", False)) or bool(
            getattr(parameter, "deprecated", False)
        ):
            return None
        # A repeated option has one static parameter identity. Its individual values are never
        # represented, including when the option happens to use a safe Choice type.
        multiple = bool(getattr(parameter, "multiple", False))
        is_bool_flag = bool(getattr(parameter, "is_bool_flag", False))
        parameter_type = getattr(parameter, "type", None)
    except BaseException:
        return None

    if is_bool_flag:
        if name in _SAFE_BOOL_PARAMETER_NAMES and type(value) is bool:
            return {"name": name, "bool_value": value}
        return {"name": name}

    if isinstance(parameter_type, click.Choice):
        allowed = _SAFE_CHOICE_VALUES.get(name)
        if allowed is not None and not multiple and isinstance(value, str) and value in allowed:
            return {"name": name, "choice_value": value}
        # An unallowlisted Choice is handled like free text: its static presence may be useful,
        # while its value is never sent.
        return {"name": name}

    bounds = _SAFE_BOUNDED_INT_RANGES.get(name)
    if bounds is not None:
        if (
            isinstance(parameter_type, click.IntRange)
            and type(value) is int
            and bounds[0] <= value <= bounds[1]
        ):
            return {"name": name, "bounded_int_value": value}
        return {"name": name}

    # Arbitrary strings, paths, IDs, secrets, JSON, and unallowlisted numeric values are presence
    # signals only. The caller has already excluded defaults and environment values.
    return {"name": name}


def _collect_parameters(ctx: Any) -> list[dict[str, Any]]:
    """Collect bounded Click option metadata without reading raw invocation text."""

    try:
        contexts = _context_chain(ctx)
        command_path = _static_command_path(contexts)
        if not contexts or command_path is None:
            return []
        entries: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for context_index, context in enumerate(contexts):
            command = getattr(context, "command", None)
            parameters = getattr(command, "params", None)
            if not isinstance(parameters, (list, tuple)):
                continue
            context_command = _static_command_path(contexts[: context_index + 1])
            if context_command is None:
                return []
            for parameter in parameters:
                # Positional arguments can carry paths, IDs, and other arbitrary user data. The
                # telemetry contract records static option names only.
                if not isinstance(parameter, click.Option):
                    continue
                parameter_name = getattr(parameter, "name", None)
                if not isinstance(parameter_name, str) or not _is_static_name(parameter_name):
                    continue
                name = f"{context_command}.{parameter_name}"
                if len(name) > _MAX_STATIC_NAME_LENGTH or name in seen_names:
                    continue
                if not _supplied_on_command_line(_parameter_source(context, parameter_name)):
                    continue
                has_value, value = _parameter_value(context, parameter_name)
                if not has_value:
                    continue
                entry = _parameter_entry(name=name, parameter=parameter, value=value)
                if entry is None:
                    continue
                entries.append(entry)
                seen_names.add(name)
                if len(entries) >= _MAX_PARAMETER_ENTRIES:
                    return entries
        return entries
    except BaseException:
        # Telemetry must fail closed if Click or an extension gives us an unexpected object shape.
        return []


def build_log(
    ctx: Any,
    *,
    execution_time_ms: int = 0,
    exit_code: int,
    error_category: str | None = None,
) -> dict[str, Any]:
    """Build the allowlisted snake_case ``AgentBricksCliLog`` JSON object."""

    command_path = _static_command_path(_context_chain(ctx)) or "agentbricks"
    try:
        duration_ms = max(0, int(execution_time_ms))
    except (TypeError, ValueError, OverflowError):
        duration_ms = 0
    try:
        command_exit_code = int(exit_code)
    except (TypeError, ValueError, OverflowError):
        command_exit_code = 1
    execution_context: dict[str, Any] = {
        "command_path": command_path,
        "package_version": _package_version(),
        "operating_system": platform.system().lower(),
        "execution_time_ms": duration_ms,
        "exit_code": command_exit_code,
    }
    if command_exit_code != 0:
        execution_context["error_category"] = _bounded_error_category(error_category)
    log: dict[str, Any] = {"execution_context": execution_context}
    parameters = _collect_parameters(ctx)
    if parameters:
        log["parameters"] = parameters
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
    execution_time_ms: int = 0,
    exit_code: int,
    error_category: str | None = None,
) -> None:
    """Emit one command record in a daemon worker, bounded by a short foreground join."""

    if telemetry_disabled():
        return
    log = build_log(
        ctx,
        execution_time_ms=execution_time_ms,
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
