"""Workspace authentication for Agent Bricks commands: validation, the preflight, and sign-in.

Which profile a command uses, and what the local config says about it, live in
`projects.profiles`; this is where a profile meets the workspace. The service never reads the
terminal: callers say whether a person is there to complete a browser login.
"""

from __future__ import annotations

import enum
import threading
from typing import Optional

from databricks_agentbricks.clients.cached_browser_auth import cached_browser_client
from databricks_agentbricks.clients.databricks_cli import run_databricks_login
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.profiles import (
    auth_login_command,
    auth_switch_command,
    sdk_auth_type,
)
from databricks_agentkit._api_client import _AgentBricksApiClient


def _validate_profile(profile: Optional[str]) -> tuple[_AgentBricksApiClient, str]:
    client = (
        cached_browser_client(profile)
        if sdk_auth_type(profile) == "external-browser"
        else _AgentBricksApiClient(profile)
    )
    return client, client.current_user


class _ValidationFailure(enum.Enum):
    NOT_SIGNED_IN = "not signed in"  # a login could fix it
    UNREACHABLE = "unreachable"  # the network or the server; signing in wouldn't fix it
    OTHER = "other"


def _classify_validation_error(error: Exception) -> _ValidationFailure:
    from databricks.sdk.errors import (
        DeadlineExceeded,
        InternalError,
        TemporarilyUnavailable,
        Unauthenticated,
    )
    from requests.exceptions import ConnectionError as RequestsConnectionError

    if isinstance(
        error, (RequestsConnectionError, InternalError, TemporarilyUnavailable, DeadlineExceeded)
    ):
        return _ValidationFailure.UNREACHABLE
    # Client construction wraps every failure (e.g. an unknown profile) in AgentCliError.
    if isinstance(error, (AgentCliError, Unauthenticated)):
        return _ValidationFailure.NOT_SIGNED_IN
    return _ValidationFailure.OTHER


def _profile_label(profile: str, source: str) -> str:
    return f"Databricks profile {profile!r} (from {source})"


def _sign_in_hint(profile: Optional[str], *, needs_terminal: bool = False) -> str:
    where = " in an interactive terminal" if needs_terminal else ""
    if profile:
        return f"Run `{auth_login_command(profile)}`{where} to sign in, then rerun the command."
    return (
        f"Run `{auth_login_command(None)}`{where} to sign in, then pass `-p <profile>` or "
        f"choose one for the project with `{auth_switch_command(None)}` (or "
        "DATABRICKS_CONFIG_PROFILE), and rerun the command."
    )


def _not_signed_in_error(profile: Optional[str], source: str) -> AgentCliError:
    if profile:
        return AgentCliError(
            f"{_profile_label(profile, source)} isn't authenticated.", hint=_sign_in_hint(profile)
        )
    return AgentCliError(
        "No Databricks profile is configured and the Databricks SDK's default authentication "
        "didn't authenticate.",
        hint=_sign_in_hint(None),
    )


def _unreachable_error(profile: Optional[str], source: str, error: Exception) -> AgentCliError:
    target = f"profile {profile!r} (from {source})" if profile else "the default auth"
    return AgentCliError(
        f"Could not reach the Databricks workspace for {target}: {error}",
        hint="Check the network (VPN/proxy) and the profile's host, then rerun the command.",
    )


# A preflight auth check must give up long before a user could plausibly complete a browser login:
# its job is to fail fast, not to host the login. `auth login` is the interactive path.
_PREFLIGHT_TIMEOUT_S = 20.0


def _validate_bounded(
    profile: Optional[str],
) -> tuple[Optional[tuple[_AgentBricksApiClient, str]], Optional[Exception]]:
    """`_validate_profile` on a daemon worker thread, giving up after `_PREFLIGHT_TIMEOUT_S`.

    Auth/network resolution can block, so the call is bounded. External-browser profiles use
    cached credentials only and never start consent. Daemon because executor threads are joined
    at interpreter exit and a hung request must not block the CLI's exit too.
    Returns (result, None) on success and (None, error) on failure or timeout.
    """
    outcome: list[tuple[Optional[tuple[_AgentBricksApiClient, str]], Optional[Exception]]] = []

    def _validate() -> None:
        try:
            outcome.append((_validate_profile(profile), None))
        except Exception as exc:  # carried to the caller, never re-raised raw
            outcome.append((None, exc))

    worker = threading.Thread(target=_validate, daemon=True, name="agentbricks-auth-preflight")
    worker.start()
    worker.join(_PREFLIGHT_TIMEOUT_S)
    if not outcome:
        return None, TimeoutError(f"no response within {_PREFLIGHT_TIMEOUT_S:g}s")
    return outcome[0]


def preflight_auth(profile: Optional[str], source: str) -> _AgentBricksApiClient:
    """Fail fast when the resolved profile can't authenticate, before a command spins anything up.

    Returns the validated client so the command can reuse it. Without this, an unauthenticated
    `external-browser` profile hangs on a browser OAuth flow that never completes — the local
    agent sits stuck with no error. The validation call itself can hang that way, so it runs
    bounded (see `_validate_bounded`).
    """
    client_and_user, error = _validate_bounded(profile)
    if client_and_user is not None:
        return client_and_user[0]
    assert error is not None
    if _classify_validation_error(error) is _ValidationFailure.UNREACHABLE:
        raise _unreachable_error(profile, source, error) from error
    raise _not_signed_in_error(profile, source) from error


def authenticate_profile(
    profile: str, source: str, *, interactive: bool
) -> tuple[_AgentBricksApiClient, str]:
    """Validate `profile`, signing in through `databricks auth login` once if it isn't yet.

    Returns the client and the signed-in user; `source` is only named in errors. Without
    `interactive` (no terminal, or CI) nobody can complete a browser login, so an unauthenticated
    profile fails with a hint instead of opening one.
    """
    label = _profile_label(profile, source)
    client_and_user, error = _validate_bounded(profile)
    if client_and_user is not None:
        return client_and_user
    assert error is not None
    failure = _classify_validation_error(error)
    if failure is _ValidationFailure.UNREACHABLE:
        raise _unreachable_error(profile, source, error) from error
    if failure is _ValidationFailure.OTHER:
        # A timeout or an unexpected failure won't be fixed by a login; don't open one.
        raise AgentCliError(f"Could not validate {label}: {error}") from error
    if not interactive:
        raise AgentCliError(
            f"Could not validate {label}: {error}",
            hint=_sign_in_hint(profile, needs_terminal=True),
        ) from error
    run_databricks_login(profile)
    client_and_user, error = _validate_bounded(profile)
    if client_and_user is None:
        raise AgentCliError(
            f"Databricks login completed, but {label} could not be validated: {error}"
        ) from error
    return client_and_user
