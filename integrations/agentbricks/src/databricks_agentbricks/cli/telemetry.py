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
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource

_DISABLE_TELEMETRY_ENV = "AGENTBRICKS_DISABLE_TELEMETRY"
_TELEMETRY_PATH = "/telemetry-ext"

# Keep the foreground wait short.  The worker is daemonized so an unavailable endpoint cannot hold
# the CLI process open after this bound expires.
_MAX_FOREGROUND_WAIT_S = 0.25
_CHOICE_VALUE_PREFIX = "AGENTBRICKS_CLI_"
_MAX_CHOICE_VALUE_LENGTH = 64

# Keep integer telemetry explicitly bounded even when the Click range later changes. This is a
# privacy review boundary, rather than an inferred copy of the command's type declaration.
_SAFE_BOUNDED_INT_RANGES: dict[str, tuple[int, int]] = {
    "agentbricks deploy.instance_count": (1, 5),
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


@dataclass(frozen=True)
class _ParameterEntry:
    name: str
    choice_value: str | None = None
    bool_value: bool | None = None
    int_value: int | None = None

    @classmethod
    def from_option(cls, name: str, option: click.Option, value: object) -> _ParameterEntry | None:
        if option.hidden or option.deprecated:
            return None
        if option.is_bool_flag:
            if name in _SAFE_BOOL_PARAMETER_NAMES and type(value) is bool:
                return cls(name=name, bool_value=value)
            return cls(name=name)
        if isinstance(option.type, click.Choice):
            if (
                not option.multiple
                and type(value) is str
                and value in option.type.choices
                and 0 < len(value) <= _MAX_CHOICE_VALUE_LENGTH
                and value.isascii()
                and all(character.isalnum() or character in "-_" for character in value)
            ):
                label = value.upper().replace("-", "_")
                return cls(name=name, choice_value=f"{_CHOICE_VALUE_PREFIX}{label}")
            return cls(name=name)
        bounds = _SAFE_BOUNDED_INT_RANGES.get(name)
        if bounds is not None and isinstance(option.type, click.IntRange):
            if type(value) is int and bounds[0] <= value <= bounds[1]:
                return cls(name=name, int_value=value)
        return cls(name=name)

    def to_dict(self) -> dict[str, str | bool | int]:
        data: dict[str, str | bool | int] = {"name": self.name}
        if self.choice_value is not None:
            data["choice_value"] = self.choice_value
        if self.bool_value is not None:
            data["bool_value"] = self.bool_value
        if self.int_value is not None:
            data["int_value"] = self.int_value
        return data


@dataclass(frozen=True)
class _AgentBricksCliLog:
    command_path: str
    package_version: str
    operating_system: str
    execution_time_ms: int
    exit_code: int
    error_category: str | None = None
    parameters: tuple[_ParameterEntry, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "command_path": self.command_path,
            "package_version": self.package_version,
            "operating_system": self.operating_system,
            "execution_time_ms": self.execution_time_ms,
            "exit_code": self.exit_code,
        }
        if self.error_category is not None:
            data["error_category"] = self.error_category
        if self.parameters:
            data["parameters"] = [parameter.to_dict() for parameter in self.parameters]
        return data


@dataclass(frozen=True)
class _FrontendLogPayload:
    log: _AgentBricksCliLog
    event_id: str
    upload_time_ms: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "frontend_log_event_id": self.event_id,
                "entry": {"agentbricks_cli_log": self.log.to_dict()},
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_body(self) -> dict[str, Any]:
        return {"uploadTime": self.upload_time_ms, "items": [], "protoLogs": [self.to_json()]}


def emit_command(
    ctx: Any,
    *,
    execution_time_ms: int = 0,
    exit_code: int,
    error_category: str | None = None,
) -> None:
    """Emit one command record in a daemon worker, bounded by a short foreground join."""

    if _telemetry_disabled():
        return
    log = _build_log(
        ctx,
        execution_time_ms=execution_time_ms,
        exit_code=exit_code,
        error_category=error_category,
    )
    payload = _FrontendLogPayload(
        log=log,
        event_id=str(uuid.uuid4()),
        upload_time_ms=int(time.time() * 1000),
    ).to_body()
    worker = threading.Thread(
        target=_send, args=(ctx, payload), name="agentbricks-telemetry", daemon=True
    )
    worker.start()
    worker.join(_MAX_FOREGROUND_WAIT_S)


def _telemetry_disabled() -> bool:
    """Whether the user opted out of CLI telemetry.

    Empty, ``0``, ``false``, ``no``, and ``off`` are false.  Other non-empty values are treated as
    true so shell conventions such as ``AGENTBRICKS_DISABLE_TELEMETRY=1`` and ``=yes`` both work.
    """

    value = os.environ.get(_DISABLE_TELEMETRY_ENV, "").strip().lower()
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


@dataclass(frozen=True)
class _ClickInvocation:
    command_path: str
    parameters: tuple[_ParameterEntry, ...]

    @classmethod
    def from_context(cls, ctx: Any) -> _ClickInvocation:
        try:
            contexts = cls._contexts(ctx)
            if contexts is None:
                return cls("agentbricks", ())
            command_path = " ".join(name for _, _, name in contexts)
            if len(command_path) > _MAX_STATIC_NAME_LENGTH:
                return cls("agentbricks", ())
            return cls(command_path, cls._entries(contexts))
        except BaseException:
            return cls("agentbricks", ())

    @staticmethod
    def _contexts(ctx: Any) -> list[tuple[Any, Any, str]] | None:
        contexts: list[tuple[Any, Any, str]] = []
        current = ctx
        for _ in range(_MAX_CONTEXT_DEPTH):
            if current is None:
                break
            if any(current is context for context, _, _ in contexts):
                return None
            command = current.command
            name = command.name
            if not _is_static_name(name):
                return None
            contexts.append((current, command, name))
            current = current.parent
        else:
            return None
        contexts.reverse()
        return contexts or None

    @staticmethod
    def _entries(contexts: list[tuple[Any, Any, str]]) -> tuple[_ParameterEntry, ...]:
        entries: list[_ParameterEntry] = []
        seen_names: set[str] = set()
        path_parts: list[str] = []
        for context, command, name in contexts:
            path_parts.append(name)
            context_path = " ".join(path_parts)
            parameters = command.params
            if not isinstance(parameters, (list, tuple)) or not parameters:
                continue
            values = context.params
            if not isinstance(values, dict):
                continue
            for parameter in parameters:
                if not isinstance(parameter, click.Option):
                    continue
                parameter_name = parameter.name
                if not _is_static_name(parameter_name):
                    continue
                entry_name = f"{context_path}.{parameter_name}"
                if len(entry_name) > _MAX_STATIC_NAME_LENGTH or entry_name in seen_names:
                    continue
                if context.get_parameter_source(parameter_name) is not ParameterSource.COMMANDLINE:
                    continue
                if parameter_name not in values:
                    continue
                entry = _ParameterEntry.from_option(entry_name, parameter, values[parameter_name])
                if entry is None:
                    continue
                entries.append(entry)
                seen_names.add(entry_name)
                if len(entries) >= _MAX_PARAMETER_ENTRIES:
                    return tuple(entries)
        return tuple(entries)


def _build_log(
    ctx: Any,
    *,
    execution_time_ms: int = 0,
    exit_code: int,
    error_category: str | None = None,
) -> _AgentBricksCliLog:
    """Build a bounded log from static Click metadata and command outcome."""

    invocation = _ClickInvocation.from_context(ctx)
    try:
        duration_ms = max(0, int(execution_time_ms))
    except (TypeError, ValueError, OverflowError):
        duration_ms = 0
    try:
        command_exit_code = int(exit_code)
    except (TypeError, ValueError, OverflowError):
        command_exit_code = 1
    return _AgentBricksCliLog(
        command_path=invocation.command_path,
        package_version=_package_version(),
        operating_system=platform.system().lower(),
        execution_time_ms=duration_ms,
        exit_code=command_exit_code,
        error_category=_bounded_error_category(error_category) if command_exit_code else None,
        parameters=invocation.parameters,
    )


def _root_obj(ctx: Any) -> Any:
    try:
        root = ctx.find_root()
    except (AttributeError, RuntimeError):
        root = ctx
    return getattr(root, "obj", getattr(ctx, "obj", None))


def _existing_workspace_client(obj: Any) -> Any | None:
    """Return a client already constructed for the command, without constructing a new one."""

    provider = getattr(obj, "api_client_provider", None)
    private_client = getattr(provider, "_client", None)
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


def _send(ctx: Any, payload: dict[str, Any]) -> None:
    try:
        client = _workspace_client(ctx)
        if client is not None:
            api_client = getattr(client, "api_client", None)
            if api_client is not None:
                api_client.do("POST", _TELEMETRY_PATH, body=payload)
    except BaseException:
        # Telemetry is deliberately silent.  In particular, do not expose transport/auth messages
        # that could contain workspace details or make a successful CLI command appear to fail.
        return


__all__ = ["emit_command"]
