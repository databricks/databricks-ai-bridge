"""Configuration of the local dispatcher: ``<name>.toml`` (settings) and ``<name>.secrets`` (keys).

    ~/.config/databricks-anthropic/          0700  (override: DATABRICKS_ANTHROPIC_CONFIG_DIR or --config-dir)
        default.toml                         settings, no secrets
        default.secrets                      0600, JSON: environment_key, sandbox_id_key, api_key (optional)
        <environment_id>.lock                single-instance lock (lock.py)

``sandbox_id_key`` is the HMAC key that names sandboxes. ``init`` generates it once and never regenerates it:
a new key would orphan the sandboxes of resumable sessions, because their names could no longer be recomputed.

Secrets are written with ``os.open(..., O_CREAT | O_EXCL, 0o600)``. A rewrite goes through a temp file created the
same way and then ``os.replace``, so a symlink planted at the path is replaced, never written through.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from databricks_anthropic.sandbox.launcher import DEFAULT_ANTHROPIC_BASE_URL

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # ty: ignore[unresolved-import]

CONFIG_DIR_ENV = "DATABRICKS_ANTHROPIC_CONFIG_DIR"
DEFAULT_CONFIG_DIR = "~/.config/databricks-anthropic"
DEFAULT_NAME = "default"
DEFAULT_SANDBOX_PREFIX = "cmalocal-"
DEFAULT_RETENTION_DAYS = 7.0
# Settings older versions wrote that are now ignored when read (``backend``: there is only one way to run tools).
LEGACY_SETTINGS = ("backend",)
LIFECYCLES = ("ephemeral", "resumable")
MCP_SERVERS = ("genie-one",)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ENVIRONMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_CREATE_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY


class ConfigError(RuntimeError):
    """The configuration is missing, unreadable or invalid; ``str(error)`` says how to fix it."""


# ------------------------------------------------------------------ directory and names


def config_dir(override: str | None = None) -> Path:
    """The configuration directory: ``--config-dir``, else ``$DATABRICKS_ANTHROPIC_CONFIG_DIR``, else the default."""
    return Path(override or os.environ.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR).expanduser()


def ensure_config_dir(path: Path) -> Path:
    """Create the directory private (0700) if it does not exist yet."""
    existed = path.exists()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not existed:
        os.chmod(path, 0o700)  # mkdir's mode is masked by the umask
    return path


def check_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise ConfigError(f"invalid config name {name!r}: use letters, digits, '-' or '_' (max 64)")
    return name


def check_environment_id(value: str) -> str:
    if not _ENVIRONMENT_ID_RE.match(value or ""):
        raise ConfigError(
            f"invalid environment id {value!r} (expected the env_... id from the Console)"
        )
    return value


def check_base_url(url: str) -> str:
    # The worker sends the session's token to this URL: never over plain HTTP.
    if not url.startswith("https://"):
        raise ConfigError(f"anthropic_base_url must use https, not {url!r}")
    return url


def paths(directory: Path, name: str) -> tuple[Path, Path]:
    """(settings file, secrets file) for a config name."""
    check_name(name)
    return directory / f"{name}.toml", directory / f"{name}.secrets"


# ------------------------------------------------------------------ settings


@dataclass
class Settings:
    """Non-secret configuration. Flags override these values for one run (see :func:`apply_overrides`)."""

    profile: str | None = None
    environment_id: str | None = None
    mcp_servers: list[str] = field(default_factory=list)
    lifecycle: str = "ephemeral"
    retention_days: float = DEFAULT_RETENTION_DAYS
    export_volume: str | None = None
    sandbox_prefix: str = DEFAULT_SANDBOX_PREFIX
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL

    def validate(self) -> None:
        if self.lifecycle not in LIFECYCLES:
            raise ConfigError(f"lifecycle must be one of {LIFECYCLES}, not {self.lifecycle!r}")
        for server in self.mcp_servers:
            if server not in MCP_SERVERS:
                raise ConfigError(
                    f"unknown MCP server {server!r} (supported: {', '.join(MCP_SERVERS)})"
                )
        if not (isinstance(self.retention_days, (int, float)) and self.retention_days >= 0):
            raise ConfigError(f"retention_days must be a number >= 0, not {self.retention_days!r}")
        if self.environment_id is not None:
            check_environment_id(self.environment_id)
        check_base_url(self.anthropic_base_url)

    def to_toml(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "environment_id": self.environment_id,
            "mcp_servers": list(self.mcp_servers),
            "lifecycle": self.lifecycle,
            "retention_days": float(self.retention_days),
            "export_volume": self.export_volume,
            "sandbox_prefix": self.sandbox_prefix,
            "anthropic_base_url": self.anthropic_base_url,
        }


_SETTING_TYPES: dict[str, type | tuple[type, ...]] = {
    "profile": str,
    "environment_id": str,
    "mcp_servers": list,
    "lifecycle": str,
    "retention_days": (int, float),
    "export_volume": str,
    "sandbox_prefix": str,
    "anthropic_base_url": str,
}


def settings_from_toml(data: Mapping[str, object]) -> Settings:
    data = {key: value for key, value in data.items() if key not in LEGACY_SETTINGS}
    unknown = sorted(set(data) - set(_SETTING_TYPES))
    if unknown:
        raise ConfigError(f"unknown setting(s) in config: {', '.join(unknown)}")
    values: dict[str, object] = {}
    for key, value in data.items():
        expected = _SETTING_TYPES[key]
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ConfigError(f"setting {key!r} has the wrong type ({type(value).__name__})")
        if key == "mcp_servers":
            if not all(isinstance(item, str) for item in value):  # type: ignore[attr-defined]
                raise ConfigError("mcp_servers must be a list of strings")
            value = list(value)  # type: ignore[arg-type]
        values[key] = value
    settings = Settings(**values)  # type: ignore[arg-type]
    settings.validate()
    return settings


def load_settings(path: Path) -> Settings:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} is not valid TOML: {e}") from e
    return settings_from_toml(data)


def save_settings(path: Path, settings: Settings) -> None:
    settings.validate()
    _write_private(path, dumps_toml(settings.to_toml()))


def apply_overrides(settings: Settings, **overrides: object) -> Settings:
    """Flags win over the file: an override that is None (flag not given) keeps the file's value."""
    changes: dict[str, Any] = {key: value for key, value in overrides.items() if value is not None}
    if "mcp_servers" in changes:
        changes["mcp_servers"] = list(changes["mcp_servers"])
    result = replace(settings, **changes)
    result.validate()
    return result


# ------------------------------------------------------------------ TOML writer (tomllib only reads)

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def toml_string(value: str) -> str:
    """A TOML basic string. Control characters become \\uXXXX escapes."""
    value.encode("utf-8")  # raises for lone surrogates, which TOML can't carry
    out = []
    for ch in value:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"TOML cannot store {value!r}")
        return repr(value)
    if isinstance(value, str):
        return toml_string(value)
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        items: list[str] = [str(item) for item in value]
        return "[" + ", ".join(toml_string(item) for item in items) + "]"
    raise TypeError(
        f"TOML writer supports strings, numbers, bools and lists of strings, not {type(value).__name__}"
    )


def dumps_toml(values: Mapping[str, object]) -> str:
    """A flat TOML document. None values are omitted (TOML has no null)."""
    lines = []
    for key, value in values.items():
        if value is None:
            continue
        if not _BARE_KEY_RE.match(key):
            raise ValueError(f"TOML key {key!r} needs quoting; only bare keys are written")
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ secrets


@dataclass
class Secrets:
    """The keys. ``repr=False`` keeps them out of tracebacks and debug output."""

    environment_key: str | None = field(default=None, repr=False)
    sandbox_id_key: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)


_SECRET_KEYS = ("environment_key", "sandbox_id_key", "api_key")


def new_sandbox_id_key() -> str:
    return secrets.token_hex(32)


def secrets_mode_problem(path: Path) -> str | None:
    """Why the secrets file is not private (any group or other bits), or None when it is."""
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return f"{path} is readable by other users (mode {mode:04o}); fix it with: chmod 600 {path}"
    return None


def load_secrets(path: Path, *, check_mode: bool = True) -> Secrets:
    """Keys from the secrets file; an absent file is an empty :class:`Secrets`."""
    if not path.exists():
        return Secrets()
    if check_mode and (problem := secrets_mode_problem(path)):
        raise ConfigError(problem)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict) or not all(
        value is None or isinstance(value, str) for value in data.values()
    ):
        raise ConfigError(f"{path} must be a JSON object of strings")
    return Secrets(**{key: data.get(key) or None for key in _SECRET_KEYS})


def save_secrets(path: Path, keys: Secrets) -> None:
    body = {key: getattr(keys, key) for key in _SECRET_KEYS if getattr(keys, key)}
    _write_private(path, json.dumps(body, indent=2, sort_keys=True) + "\n")


# ------------------------------------------------------------------ private file writes


def _write_private(path: Path, text: str) -> None:
    """Create ``path`` mode 0600, or replace it safely if it exists."""
    data = text.encode("utf-8")
    try:
        fd = os.open(path, _CREATE_FLAGS, 0o600)
    except FileExistsError:
        _replace_private(path, data)
        return
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise


def _replace_private(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp, _CREATE_FLAGS, 0o600)  # O_EXCL: never reuses or follows an existing file
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # replaces a symlink at `path` itself, never its target
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
