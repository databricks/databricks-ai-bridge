"""Databricks login for the local dispatcher: a named Databricks CLI profile, nothing else.

The sandbox identity is whoever runs the dispatcher, so this login decides whose permissions every session on the
environment gets. The only supported login is a profile created by ``databricks auth login`` (OAuth U2M, or a
service principal chosen at sign-in): the CLI keeps the one-hour OAuth tokens fresh, and the SDK reads them through
``databricks auth token``. The CLI must be installed.

Nothing here picks a profile for you. The SDK would otherwise fall back to ``DEFAULT`` or to
``DATABRICKS_CONFIG_PROFILE``, so a missing profile is an error with instructions, never a silent choice.

This module never prompts, prints or installs anything, and it never mutates ``os.environ``. Callers decide how to
show :class:`AuthSetupError` messages.
"""

from __future__ import annotations

import configparser
import os
import platform
import re
import shutil
import subprocess

from databricks.sdk import WorkspaceClient

from databricks_anthropic._useragent import PRODUCT_NAME, PRODUCT_VERSION, subprocess_env

MIN_CLI_VERSION = (0, 205, 0)  # first release of the current (Go) CLI with `databricks auth login`
CLI_AUTH_TYPE = "databricks-cli"
WORKSPACE_URL_PLACEHOLDER = "https://<your-workspace-url>"


class AuthSetupError(RuntimeError):
    """Databricks login is not usable yet; ``str(error)`` holds step-by-step instructions."""


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
    """Settings of a named profile in ~/.databrickscfg. No name, no profile: ``[DEFAULT]`` is never read here."""
    if not profile:
        return {}
    path = os.environ.get("DATABRICKS_CONFIG_FILE", os.path.expanduser("~/.databrickscfg"))
    parser = configparser.ConfigParser(interpolation=None)  # tokens may contain '%'
    parser.read(path)
    return dict(parser[profile]) if parser.has_section(profile) else {}


# ------------------------------------------------------------------ instructions


def login_instructions(*, reason: str, profile: str | None = None, host: str | None = None) -> str:
    """How to create (or refresh) a Databricks CLI profile. No OAuth app is needed."""
    install = "\n".join(f"      {c}" for c in cli_install_commands())
    url = host or WORKSPACE_URL_PLACEHOLDER
    name = profile or "<profile-name>"
    return (
        f"{reason}\n\n"
        "Sign in with the Databricks CLI (OAuth; you don't need to configure an OAuth app):\n"
        f"  1. Install the Databricks CLI (v{'.'.join(map(str, MIN_CLI_VERSION))} or newer), for example:\n{install}\n"
        "  2. Copy your workspace URL from the browser address bar while you're in the workspace. Keep the\n"
        "     https:// and leave off anything after the domain.\n"
        f"  3. Run:  databricks auth login --host {url}\n"
        "     The CLI opens a browser window: sign in to the workspace and approve access. If you can act as a\n"
        "     service principal, you'll also be asked whether to sign in as yourself or as that service principal.\n"
        f"  4. Back in the terminal, keep the suggested profile name or type your own (for example {name}).\n"
        "     The CLI saves the workspace URL under that profile and refreshes the one-hour OAuth tokens for you.\n"
        f"  5. Use it here:  --profile {name}\n"
        "  To sign in again later, or to add another workspace, run step 3 again with that workspace's URL.\n"
    )


def no_profile_instructions() -> str:
    return login_instructions(
        reason=(
            "No Databricks profile given. Pass --profile <name>, a profile created by `databricks auth login`.\n"
            "The dispatcher never picks a profile for you (not DEFAULT, not DATABRICKS_CONFIG_PROFILE), because\n"
            "every session on the environment runs as that identity."
        )
    )


# ------------------------------------------------------------------ resolution


def check_profile(profile: str | None) -> str:
    """The profile's workspace host, after checking it's a usable CLI profile; raise :class:`AuthSetupError`."""
    if not profile:
        raise AuthSetupError(no_profile_instructions())
    settings = read_profile(profile)
    host = settings.get("host")
    if not settings:
        raise AuthSetupError(
            login_instructions(
                reason=f"Profile {profile!r} not found in ~/.databrickscfg.", profile=profile
            )
        )
    if settings.get("auth_type") != CLI_AUTH_TYPE:
        raise AuthSetupError(
            login_instructions(
                reason=(
                    f"Profile {profile!r} is not a Databricks CLI login (auth_type is "
                    f"{settings.get('auth_type') or 'not set'}). The local dispatcher only uses profiles created "
                    "by `databricks auth login`."
                ),
                profile=profile,
                host=host,
            )
        )
    path = cli_path()
    if not path:
        raise AuthSetupError(
            login_instructions(
                reason=f"Profile {profile!r} needs the Databricks CLI, which is not installed.",
                profile=profile,
                host=host,
            )
        )
    version = cli_version(path)
    if version and version < MIN_CLI_VERSION:
        raise AuthSetupError(
            login_instructions(
                reason=(
                    f"Databricks CLI {'.'.join(map(str, version))} at {path} is the legacy CLI; "
                    "upgrade to the current CLI."
                ),
                profile=profile,
                host=host,
            )
        )
    return host or ""


def workspace_client(profile: str) -> WorkspaceClient:
    """A client for a named CLI profile. The environment is read, never written."""
    return WorkspaceClient(profile=profile, product=PRODUCT_NAME, product_version=PRODUCT_VERSION)


def login(*, profile: str | None) -> WorkspaceClient:
    """Check the CLI profile and return a verified WorkspaceClient; raise :class:`AuthSetupError` with fix-it steps."""
    host = check_profile(profile)
    assert profile  # check_profile raised otherwise
    try:
        w = workspace_client(profile)
        w.current_user.me()
    except Exception as e:  # noqa: BLE001 - expired or revoked login, wrong host, ...
        raise AuthSetupError(
            login_instructions(
                reason=f"Login for profile {profile!r} failed ({type(e).__name__}); sign in again.",
                profile=profile,
                host=host or None,
            )
        ) from e
    return w
