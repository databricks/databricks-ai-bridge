"""Local Databricks profile configuration: which profile a command uses, and what the config says.

Everything here reads local files and the environment only; it never contacts a workspace.
"""

from __future__ import annotations

import configparser
import dataclasses
import json
import os
import pathlib
from typing import Optional

from databricks_agentbricks.projects.env_file import read_env

# Where a resolved profile came from; the strings are user-facing too.
SOURCE_FLAG = "--profile"
SOURCE_ENV_FILE = ".env"
SOURCE_ENV_VAR = "DATABRICKS_CONFIG_PROFILE"
SOURCE_SDK_DEFAULT = "Databricks SDK default"
SOURCE_SETTINGS_DEFAULT = "databricks auth switch"
SOURCE_CONFIG_DEFAULT = "default"

_SETTINGS_SECTION = "__settings__"
# Any of these without a profile means the SDK authenticates from env vars and never reads the
# config file.
_ENV_AUTH_VARS = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID")


def auth_login_command(profile: Optional[str]) -> str:
    return f"agentbricks auth login {profile or '<profile>'}"


def auth_switch_command(profile: Optional[str]) -> str:
    return f"agentbricks auth switch {profile or '<profile>'}"


@dataclasses.dataclass(frozen=True)
class ProfileInfo:
    """The profile a command runs with, and where it came from (for display and re-resolution)."""

    name: Optional[str]
    source: str


def resolve_profile(
    flag: Optional[str],
    project_dir: Optional[pathlib.Path] = None,
    env_values: Optional[dict[str, str]] = None,
) -> ProfileInfo:
    """Pick the Databricks profile for a command, most to least specific.

    1. the `-p/--profile` flag,
    2. the project `.env`'s `DATABRICKS_CONFIG_PROFILE` (only when `project_dir` is given, so the CLI
       and the locally running agent, which reads the same `.env`, agree on one profile),
    3. the `DATABRICKS_CONFIG_PROFILE` environment variable,
    4. none — the Databricks SDK's own default authentication.

    ``env_values`` lets a caller that already parsed ``<project_dir>/.env`` pass it in instead of
    having it read again.
    """
    if flag:
        return ProfileInfo(flag, SOURCE_FLAG)
    if project_dir is not None:
        if env_values is None:
            env_values = read_env(project_dir / ".env")
        file_profile = env_values.get("DATABRICKS_CONFIG_PROFILE")
        if file_profile:
            return ProfileInfo(file_profile, SOURCE_ENV_FILE)
    env_profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if env_profile:
        return ProfileInfo(env_profile, SOURCE_ENV_VAR)
    return ProfileInfo(None, SOURCE_SDK_DEFAULT)


def login_target(
    name: Optional[str], flag: ProfileInfo, env_profile: Optional[str]
) -> Optional[tuple[str, str]]:
    """(profile, source) for `auth login`: NAME, the global `-p`, or the project's `.env` only.

    Signing in is an explicit act, so the ambient DATABRICKS_CONFIG_PROFILE and the SDK default
    are deliberately not consulted. None means nothing chose one.
    """
    if name:
        return name, "argument"
    if flag.source == SOURCE_FLAG and flag.name:
        return flag.name, SOURCE_FLAG
    if env_profile:
        return env_profile, SOURCE_ENV_FILE
    return None


def _config_path() -> pathlib.Path:
    # expanduser: the SDK expands `~` in DATABRICKS_CONFIG_FILE; without it a tilde path reads nothing.
    return pathlib.Path(
        os.getenv("DATABRICKS_CONFIG_FILE", str(pathlib.Path.home() / ".databrickscfg"))
    ).expanduser()


_config_cache: Optional[tuple[tuple[str, int, int], dict[str, dict[str, str]]]] = None


def _config_sections() -> dict[str, dict[str, str]]:
    """Every section's own keys, `[DEFAULT]` included as an ordinary section; never raises.

    Sections don't inherit `[DEFAULT]`, matching how the SDK reads a named profile. No
    interpolation, so a `%` in a value is taken verbatim. Cached on the file's identity, since one
    command looks the config up several times and tests rewrite it between commands.
    """
    global _config_cache
    path = _config_path()
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _config_cache is not None and _config_cache[0] == key:
        return _config_cache[1]
    parser = configparser.ConfigParser(interpolation=None, default_section="\0no-fallback")
    try:
        parser.read(path)
    except (OSError, configparser.Error):
        pass  # keep whatever parsed before the problem, as the SDK-facing lookups always have
    sections = {name: dict(parser[name]) for name in parser.sections()}
    _config_cache = (key, sections)
    return sections


def _profile_section(profile: str) -> dict[str, str]:
    return _config_sections().get(profile, {})


def profile_exists(profile: str) -> bool:
    """Whether the profile is configured in the Databricks config file; never raises.

    `[DEFAULT]` counts only once it has a key, the way the SDK falls back to it.
    """
    sections = _config_sections()
    if profile == configparser.DEFAULTSECT:
        return bool(sections.get(profile))
    return profile in sections


def profile_host(profile: Optional[str]) -> Optional[str]:
    """Host configured for `profile` in the Databricks config file; None when unknown. Never raises."""
    if not profile:
        return None
    return _profile_section(profile).get("host")


def configured_profiles() -> list[str]:
    """Profile names in the Databricks config file, `[DEFAULT]` first when it exists."""
    names = [
        name
        for name in _config_sections()
        if name not in (_SETTINGS_SECTION, configparser.DEFAULTSECT)
    ]
    return ([configparser.DEFAULTSECT] if profile_exists(configparser.DEFAULTSECT) else []) + names


def settings_default_profile() -> Optional[str]:
    """The default profile chosen with `databricks auth switch` ([__settings__] default_profile)."""
    return _profile_section(_SETTINGS_SECTION).get("default_profile", "").strip() or None


def env_auth_configured() -> bool:
    """Whether env-var auth is set, so without a profile the SDK never reads the config file."""
    return any(os.environ.get(name) for name in _ENV_AUTH_VARS)


def sdk_default_profile() -> tuple[str, str]:
    """(profile, source) the SDK falls back to without one: the switched default, else DEFAULT.

    Only meaningful when `env_auth_configured()` is False; otherwise the config file isn't read.
    """
    switched = settings_default_profile()
    if switched:
        return switched, SOURCE_SETTINGS_DEFAULT
    return configparser.DEFAULTSECT, SOURCE_CONFIG_DEFAULT


def workspace_for_profile(profile: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """An offline workspace estimate as (host, display text), for notices and `auth describe`.

    An exported host wins even with an explicit profile. Without a profile, common environment
    credentials suppress the file; otherwise prefer the switched default, then `[DEFAULT]`.
    Callers with a constructed SDK client should use its resolved host instead.
    """
    env_host = os.environ.get("DATABRICKS_HOST")
    if env_host:
        return env_host, f"{env_host} (from DATABRICKS_HOST)"
    if profile:
        host = profile_host(profile)
        return host, host
    if env_auth_configured():
        return None, "none configured"
    default_profile, _ = sdk_default_profile()
    default_host = profile_host(default_profile)
    if default_host:
        return default_host, f"{default_host} ([{default_profile}] profile)"
    return None, "none configured"


def legacy_default_profile() -> Optional[str]:
    """The default profile the removed `agentbricks login` saved, if its file is still around."""
    base = os.environ.get("AGENTBRICKS_CONFIG_HOME")
    path = (pathlib.Path(base) if base else pathlib.Path.home() / ".agentbricks") / "config.json"
    try:
        return json.loads(path.read_text()).get("profile") or None
    except (OSError, ValueError, AttributeError):
        return None


def sdk_auth_type(profile: Optional[str]) -> Optional[str]:
    """The auth_type the Databricks SDK would use for `profile`, resolved the way the SDK does."""
    env_auth_type = os.environ.get("DATABRICKS_AUTH_TYPE")
    if env_auth_type:
        return env_auth_type
    if not profile:
        if env_auth_configured():
            return None
        profile, _ = sdk_default_profile()
    return _profile_section(profile).get("auth_type")
