"""Databricks U2M auth for the dispatcher: Databricks CLI profiles or SDK browser login.

The sandbox identity is whoever creates the sandbox, so the dispatcher's Databricks
login decides whose permissions the agent (and Genie One) runs with. Two U2M paths:

- ``cli``: a CLI profile (``auth_type = databricks-cli``, created by
  ``databricks auth login``). The SDK shells out to ``databricks auth token``, so the
  Databricks CLI must be installed on the dispatcher host.
- ``browser``: the SDK's own OAuth login (``auth_type = external-browser``); no CLI.
  Uses your OAuth app when ``client_id`` is configured (it must allow the redirect URL
  ``http://localhost:8020``, which the SDK hard-codes), otherwise Databricks' built-in
  public ``databricks-cli`` client. Tokens are cached by the SDK.

The default is ``cli``. ``auto`` picks ``cli`` when the profile is a CLI profile and the
CLI is installed, else ``browser`` when a host is known. Profiles of other types (PAT,
OAuth M2M, ...) are handed to the SDK as-is. Anything missing raises
:class:`AuthSetupError` with instructions. This module never prompts, prints or installs
anything; callers decide how to show the instructions.
"""

from __future__ import annotations

import configparser
import contextlib
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient

from databricks_anthropic._useragent import (
    PRODUCT_NAME,
    PRODUCT_VERSION,
    apply_process_env,
    subprocess_env,
)

SDK_BROWSER_REDIRECT_URL = "http://localhost:8020"
MIN_CLI_VERSION = (0, 205, 0)  # first release of the current (Go) CLI with `databricks auth login`
OAUTH_CLIENT_ID_ENV = "DATABRICKS_U2M_CLIENT_ID"
OAUTH_CLIENT_SECRET_ENV = "DATABRICKS_U2M_CLIENT_SECRET"


class AuthSetupError(RuntimeError):
    """Databricks auth is not usable yet; ``str(error)`` holds step-by-step instructions."""


@dataclass(frozen=True)
class AuthChoice:
    mode: str  # "cli" | "browser" | "profile" | "env"
    profile: str | None
    host: str | None
    client_id: str | None


# ------------------------------------------------------------------ CLI detection


def cli_path() -> str | None:
    return shutil.which("databricks")


def cli_version(path: str) -> tuple[int, int, int] | None:
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=20, env=subprocess_env()
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(x) for x in match.groups()) if match else None  # type: ignore[return-value]


def cli_install_commands() -> list[str]:
    """Official install commands for this OS (see docs.databricks.com/dev-tools/cli/install)."""
    system = platform.system()
    if system == "Darwin":
        return [
            "brew tap databricks/tap && brew install databricks",
            "curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh",
        ]
    if system == "Windows":
        return ["winget install Databricks.DatabricksCLI"]
    return [
        "curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sudo sh"
    ]


# ------------------------------------------------------------------ profiles


def read_profile(profile: str | None) -> dict[str, str]:
    path = os.environ.get("DATABRICKS_CONFIG_FILE", os.path.expanduser("~/.databrickscfg"))
    parser = configparser.ConfigParser(interpolation=None)  # tokens may contain '%'
    parser.read(path)
    name = profile or "DEFAULT"
    if name == "DEFAULT":
        return dict(parser.defaults())
    return dict(parser[name]) if parser.has_section(name) else {}


# ------------------------------------------------------------------ instructions


def cli_instructions(host: str | None, profile: str | None, *, reason: str) -> str:
    install = "\n".join(f"      {c}" for c in cli_install_commands())
    login_host = host or "https://<your-workspace>.cloud.databricks.com"
    return (
        f"{reason}\n\n"
        "Option A: Databricks CLI login (recommended)\n"
        f"  1. Install the Databricks CLI (v{'.'.join(map(str, MIN_CLI_VERSION))} or newer), e.g.:\n{install}\n"
        f"  2. Log in once (opens your browser):\n"
        f"      databricks auth login --host {login_host} --profile {profile or '<profile-name>'}\n"
        f"  3. Re-run with --profile {profile or '<profile-name>'}\n"
    )


def browser_instructions(host: str | None) -> str:
    return (
        "Option B: browser login without the CLI (--auth browser --host <workspace-url>)\n"
        "  Uses Databricks' built-in public OAuth client by default; nothing to configure.\n"
        "  To use your own OAuth app instead:\n"
        "    1. Account console -> Settings -> App connections -> Add connection\n"
        f"    2. Redirect URL: exactly {SDK_BROWSER_REDIRECT_URL} (fixed by the Databricks SDK)\n"
        "    3. Access scopes: all-apis (offline_access is added for refresh tokens)\n"
        "    4. Keep the client ID (and secret, for a confidential app), then set\n"
        f"         {OAUTH_CLIENT_ID_ENV}=<client-id>   {OAUTH_CLIENT_SECRET_ENV}=<secret>\n"
        "       (or pass --oauth-env-file / --oauth-client-id)\n"
        f"  Host: {host or '<workspace-url>'}\n"
    )


# ------------------------------------------------------------------ resolution


def choose_auth(
    *,
    auth: str = "cli",
    profile: str | None = None,
    host: str | None = None,
    client_id: str | None = None,
) -> AuthChoice:
    """Decide how to authenticate; raise :class:`AuthSetupError` with instructions if neither works."""
    settings = read_profile(profile)
    host = host or settings.get("host")
    client_id = client_id or os.environ.get(OAUTH_CLIENT_ID_ENV) or None
    profile_type = settings.get("auth_type")
    if profile and not settings:
        raise AuthSetupError(
            cli_instructions(
                host, profile, reason=f"Profile {profile!r} not found in ~/.databrickscfg."
            )
            + "\n"
            + browser_instructions(host)
        )

    def check_cli() -> AuthChoice:
        path = cli_path()
        if not path:
            raise AuthSetupError(
                cli_instructions(
                    host,
                    profile,
                    reason=(
                        f"Profile {profile!r} uses auth_type=databricks-cli, but the Databricks CLI is not installed."
                        if profile
                        else "The Databricks CLI is not installed."
                    ),
                )
                + "\n"
                + browser_instructions(host)
            )
        version = cli_version(path)
        if version and version < MIN_CLI_VERSION:
            raise AuthSetupError(
                cli_instructions(
                    host,
                    profile,
                    reason=(
                        f"Databricks CLI {'.'.join(map(str, version))} at {path} is the legacy CLI; "
                        "upgrade to the current CLI."
                    ),
                )
            )
        return AuthChoice("cli", profile, host, None)

    if profile and profile_type != "databricks-cli":
        # PAT, OAuth M2M, external-browser profiles, ...: the SDK handles them, no CLI needed.
        return AuthChoice("profile", profile, host, None)
    if not profile and auth in ("cli", "auto") and _sdk_env_auth():
        # Unified-auth env vars (e.g. a service principal's DATABRICKS_CLIENT_ID/SECRET): no CLI needed.
        return AuthChoice("env", None, host or os.environ.get("DATABRICKS_HOST"), None)
    if auth == "cli":
        return check_cli()
    if auth == "browser":
        if not host:
            raise AuthSetupError(
                "Browser login needs the workspace URL: pass --host.\n\n"
                + browser_instructions(host)
            )
        return AuthChoice("browser", None, host, client_id)
    # auto
    if profile_type == "databricks-cli":
        if cli_path():
            return check_cli()
        if host:  # same workspace, no CLI: fall back to the SDK's browser login
            return AuthChoice("browser", None, host, client_id)
        return check_cli()  # raises with instructions
    if host:
        return AuthChoice("browser", None, host, client_id)
    raise AuthSetupError(
        cli_instructions(None, None, reason="No Databricks login configured.")
        + "\n"
        + browser_instructions(None)
    )


def _sdk_env_auth() -> bool:
    """DATABRICKS_HOST plus a token or OAuth M2M client credentials in the environment."""
    env = os.environ
    return bool(env.get("DATABRICKS_HOST")) and bool(
        env.get("DATABRICKS_TOKEN")
        or (env.get("DATABRICKS_CLIENT_ID") and env.get("DATABRICKS_CLIENT_SECRET"))
    )


@contextlib.contextmanager
def _without_env(name: str):
    """Hide one environment variable while a client is built, then restore it."""
    saved = os.environ.pop(name, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[name] = saved


def workspace_client(choice: AuthChoice, *, client_secret: str | None = None) -> WorkspaceClient:
    # Partner attribution, set by the connector: the SDK's `databricks auth token` subprocesses inherit it.
    apply_process_env()
    if choice.mode == "browser":
        secret = client_secret or os.environ.get(OAUTH_CLIENT_SECRET_ENV) or None
        # An ambient DATABRICKS_CONFIG_PROFILE (often DEFAULT, possibly another workspace) would be
        # merged into this explicit-host config; hide it while the client is built.
        with _without_env("DATABRICKS_CONFIG_PROFILE"):
            return WorkspaceClient(
                host=choice.host,
                auth_type="external-browser",
                client_id=choice.client_id or None,
                client_secret=secret if choice.client_id else None,
                product=PRODUCT_NAME,
                product_version=PRODUCT_VERSION,
            )
    return WorkspaceClient(
        profile=choice.profile or None, product=PRODUCT_NAME, product_version=PRODUCT_VERSION
    )


def login(
    *,
    auth: str = "cli",
    profile: str | None = None,
    host: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> WorkspaceClient:
    """Resolve auth and return a verified WorkspaceClient; raise :class:`AuthSetupError` with fix-it steps."""
    choice = choose_auth(auth=auth, profile=profile, host=host, client_id=client_id)
    try:
        w = workspace_client(choice, client_secret=client_secret)
    except ValueError as e:
        raise AuthSetupError(
            (
                browser_instructions(choice.host)
                if choice.mode == "browser"
                else cli_instructions(
                    choice.host, choice.profile, reason="Databricks login failed."
                )
            )
            + f"\nSDK error: {e}"
        ) from e
    try:
        w.current_user.me()
    except Exception as e:  # noqa: BLE001
        hint = (
            cli_instructions(
                choice.host,
                choice.profile,
                reason=f"Login for profile {choice.profile!r} failed: {e}",
            )
            if choice.mode == "cli"
            else browser_instructions(choice.host)
        )
        raise AuthSetupError(hint) from e
    return w
