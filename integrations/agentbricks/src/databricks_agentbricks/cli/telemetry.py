"""Best effort Agent Bricks CLI invocation telemetry.

The CLI sends one small ``FrontendLogEntry`` payload after each leaf command.  This module keeps
the logging schema and transport isolated from command behavior: telemetry is opt-out, never
prints a diagnostic, and never changes the command's result when authentication or transport is
unavailable.
"""

from __future__ import annotations

import json
import math
import os
import platform
import threading
import time
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from typing import Any, Literal
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import click
from click.core import ParameterSource
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from databricks_agentbricks.telemetry.scrub import scrub_failure_text

_DISABLE_TELEMETRY_ENV = "AGENTBRICKS_DISABLE_TELEMETRY"
_TELEMETRY_PATH = "/telemetry-ext"

# Keep the foreground wait short.  The worker is daemonized so an unavailable endpoint cannot hold
# the CLI process open after this bound expires.
_MAX_FOREGROUND_WAIT_S = 2.0
_CHOICE_VALUE_PREFIX = "AGENTBRICKS_CLI_"

_MAX_PARAMETER_ENTRIES = 64
_MAX_CONTEXT_DEPTH = 32
_MAX_STATIC_NAME_LENGTH = 128
_DEPLOYMENT_META_KEY = "agentbricks.deployment_telemetry"
DeploymentOutcome = Literal[
    "DEPLOYMENT_OUTCOME_UNSPECIFIED",
    "DEPLOYMENT_OUTCOME_RUNNING",
    "DEPLOYMENT_OUTCOME_SUCCEEDED",
    "DEPLOYMENT_OUTCOME_FAILED",
]


@dataclass
class DeploymentTelemetry:
    """Attribution for a single printed command result, not a logical rollout."""

    invocation_id: str
    background_child: bool = False
    details_link_emitted: bool = False
    deployment_outcome: DeploymentOutcome = "DEPLOYMENT_OUTCOME_UNSPECIFIED"


def begin_deployment(*, background_child: bool = False) -> None:
    ctx = click.get_current_context()
    ctx.meta[_DEPLOYMENT_META_KEY] = DeploymentTelemetry(
        invocation_id=str(uuid.uuid4()), background_child=background_child
    )


def set_deployment_outcome(outcome: DeploymentOutcome) -> None:
    """Record observed state without replacing a deployment exception."""
    try:
        ctx = click.get_current_context(silent=True)
        attribution = _deployment_metadata(ctx)
        if attribution is not None:
            attribution.deployment_outcome = outcome
    except BaseException:
        return


def deployment_details_url(url: str | None) -> str | None:
    """Join browser arrivals to command telemetry while preserving opt-out."""
    ctx = click.get_current_context()
    attribution = ctx.meta.get(_DEPLOYMENT_META_KEY)
    if not isinstance(attribution, DeploymentTelemetry):
        return url
    attribution.details_link_emitted = url is not None
    if url is None or attribution.background_child or _telemetry_disabled():
        return url
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.extend([("source", "agentbricks-cli"), ("cli_invocation_id", attribution.invocation_id)])
    return urlunsplit(parts._replace(query=urlencode(query)))


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
    double_value: float | None = None

    @classmethod
    def from_option(cls, name: str, option: click.Option, value: object) -> _ParameterEntry:
        if isinstance(option.type, click.Choice):
            if value in option.type.choices:
                label = option.type.normalize_choice(value, None).upper().replace("-", "_")
                return cls(name=name, choice_value=f"{_CHOICE_VALUE_PREFIX}{label}")
            return cls(name=name)
        if isinstance(option.type, click.types.BoolParamType) and type(value) is bool:
            return cls(name=name, bool_value=value)
        if isinstance(option.type, click.types.IntParamType) and type(value) is int:
            if -(1 << 63) <= value < (1 << 63):
                return cls(name=name, int_value=value)
        if isinstance(option.type, click.types.FloatParamType) and type(value) is float:
            if math.isfinite(value):
                return cls(name=name, double_value=value)
        return cls(name=name)

    def to_dict(self) -> dict[str, str | bool | int | float]:
        data: dict[str, str | bool | int | float] = {"name": self.name}
        if self.choice_value is not None:
            data["choice_value"] = self.choice_value
        if self.bool_value is not None:
            data["bool_value"] = self.bool_value
        if self.int_value is not None:
            data["int_value"] = self.int_value
        if self.double_value is not None:
            data["double_value"] = self.double_value
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
    failure_message: str | None = None
    deployment: DeploymentTelemetry | None = None

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
        if self.failure_message is not None:
            data["failure_message"] = self.failure_message
        if self.deployment is not None:
            data.update(
                invocation_id=self.deployment.invocation_id,
                background_child=self.deployment.background_child,
                details_link_emitted=self.deployment.details_link_emitted,
                deployment_outcome=self.deployment.deployment_outcome,
            )
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
    exception: BaseException | None = None,
) -> None:
    """Emit one command record in a daemon worker, bounded by a short foreground join."""

    if _telemetry_disabled():
        return
    log = _build_log(
        ctx,
        execution_time_ms=execution_time_ms,
        exit_code=exit_code,
        error_category=error_category,
        exception=exception,
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


def _exception_chain(exc: BaseException | None):
    seen: set[int] = set()
    for _ in range(8):
        if exc is None or id(exc) in seen:
            return
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__


_SERVICE_ERROR_CATEGORIES = {
    code: f"ERROR_CATEGORY_{category}"
    for category, codes in {
        "AUTHENTICATION": (
            "UNAUTHENTICATED",
            "INVALID_TOKEN",
            "INVALID_CREDENTIALS",
            "TOKEN_EXPIRED",
        ),
        "AUTHORIZATION": ("PERMISSION_DENIED", "FORBIDDEN", "UNAUTHORIZED", "NOT_AUTHORIZED"),
        "NETWORK": (
            "NETWORK_ERROR",
            "CONNECTION_ERROR",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "TIMEOUT",
            "TIMED_OUT",
            "CANCELLED",
            "ABORTED",
        ),
        "CONFIGURATION": ("INVALID_ARGUMENT", "FAILED_PRECONDITION", "CONFIGURATION_ERROR"),
        "RUNTIME": (
            "NOT_FOUND",
            "ALREADY_EXISTS",
            "RESOURCE_EXHAUSTED",
            "INTERNAL_ERROR",
            "FEATURE_DISABLED",
            "UNIMPLEMENTED",
            "NOT_IMPLEMENTED",
        ),
    }.items()
    for code in codes
}


def _error_category(exc: BaseException) -> str:
    """Map an exception to the bounded enum in ``agentbricks_cli_log.proto``."""

    # Categories remain code-defined even when a separate sensitive message is available.
    import click

    if isinstance(exc, click.UsageError):
        return "ERROR_CATEGORY_USAGE"
    if isinstance(exc, (KeyboardInterrupt, click.Abort)):
        return "ERROR_CATEGORY_INTERRUPTED"
    chain = list(_exception_chain(exc))
    for error in chain:
        category = getattr(error, "error_category", None)
        if isinstance(category, str) and category in _ERROR_CATEGORIES:
            return category
        code = getattr(error, "error_code", None)
        if isinstance(code, str) and (category := _SERVICE_ERROR_CATEGORIES.get(code.upper())):
            return category
    for error in chain:
        if isinstance(error, ImportError):
            return "ERROR_CATEGORY_DEPENDENCY"
        if isinstance(
            error,
            (TimeoutError, ConnectionError, URLError, RequestsConnectionError, RequestsTimeout),
        ):
            return "ERROR_CATEGORY_NETWORK"
        if isinstance(
            error, (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)
        ):
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
        """Collect options generically from Click's parsed values, including resolved defaults."""

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
                source = context.get_parameter_source(parameter_name)
                if source not in (ParameterSource.COMMANDLINE, ParameterSource.DEFAULT):
                    continue
                if parameter_name not in values:
                    continue
                # Click has already resolved and converted defaults (including callables). Read
                # the value used by the command rather than evaluating its default a second time.
                value = values[parameter_name]
                option_values = (
                    value if parameter.multiple and isinstance(value, tuple) else (value,)
                )
                for option_value in option_values:
                    entry = _ParameterEntry.from_option(entry_name, parameter, option_value)
                    # Defaults contribute only values already supported by the bounded wire types.
                    # Do not add presence-only entries for omitted free-form or unresolved options.
                    if source is ParameterSource.DEFAULT and len(entry.to_dict()) == 1:
                        continue
                    entries.append(entry)
                    if len(entries) >= _MAX_PARAMETER_ENTRIES:
                        return tuple(entries)
                seen_names.add(entry_name)
        return tuple(entries)


def _build_log(
    ctx: Any,
    *,
    execution_time_ms: int = 0,
    exit_code: int,
    error_category: str | None = None,
    exception: BaseException | None = None,
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
    category = _bounded_error_category(error_category) if command_exit_code else None
    is_failure = bool(command_exit_code) and category != "ERROR_CATEGORY_INTERRUPTED"
    if (
        is_failure
        and invocation.command_path == "agentbricks deployments wait"
        and command_exit_code == 75
    ):
        category = None
        is_failure = False
    return _AgentBricksCliLog(
        command_path=invocation.command_path,
        package_version=_package_version(),
        operating_system=platform.system().lower(),
        execution_time_ms=duration_ms,
        exit_code=command_exit_code,
        error_category=category,
        parameters=invocation.parameters,
        failure_message=_failure_message(exception) if is_failure else None,
        deployment=_deployment_metadata(ctx),
    )


def _failure_message(exc: BaseException | None) -> str | None:
    if isinstance(
        exc, (KeyboardInterrupt, click.Abort, click.UsageError, click.exceptions.Exit, SystemExit)
    ):
        return None
    parts: list[str] = []
    for error in _exception_chain(exc):
        try:
            text = str(error)
        except Exception:
            continue
        if text and text not in parts:
            parts.append(text[:65536])
        detail = getattr(error, "failure_detail", None)
        if isinstance(detail, str) and detail and detail not in parts:
            parts.append(detail[:65536])
    return scrub_failure_text("\nCaused by: ".join(parts)) or None


def _deployment_metadata(ctx: Any) -> DeploymentTelemetry | None:
    metadata = getattr(ctx, "meta", {}).get(_DEPLOYMENT_META_KEY)
    return metadata if isinstance(metadata, DeploymentTelemetry) else None


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


def _workspace_client(ctx: Any) -> Any | None:
    obj = _root_obj(ctx)
    existing = _existing_workspace_client(obj)
    if existing is not None:
        return existing
    profile = getattr(obj, "profile", None)
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
