"""Pure deploy helpers and constants shared by the CLI and (soon) the portable deploy service.

These are the render-free, click-free pieces of ``agentbricks deploy``: deployment-name validation
and prefixing, the fixed-instance runtime args, the name/prefix/length + pip-index + rollout
constants, and the MLflow tracing env config. They live here - not in ``cli/deploy.py`` - so a
future ``services/`` module can import them without importing the Click command module (which would
create a circular import). ``cli/deploy.py`` re-imports every name from here, so its existing callers
(the ``deployments`` group, ``DeployOrchestrator``, ``endpoint.py``, and the tests) keep resolving
them unchanged.

This module is intentionally pure: it imports neither ``click`` nor ``render`` nor anything under
``cli``, so importing it never drags the CLI presentation layer into a service.
"""

from __future__ import annotations

from dataclasses import dataclass

from databricks_agentbricks.errors import AgentCliError

# TEMPORARY: the Apps build environment currently can't reach the internal pypi proxy, so builds
# time out installing dependencies. Point the build at public PyPI (sanctioned interim workaround)
# until the proxy is reachable from the build sandbox again, then drop this default. pip reads
# PIP_INDEX_URL; uv reads UV_INDEX_URL / UV_DEFAULT_INDEX — set all three to cover both build paths.
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple/"
_PIP_INDEX_ENVS = ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX")
_AGENT_COMPUTE_OUTPUT = ("App compute", "Agent compute")
# Internal rollout switch. Backend selection is intentionally not part of the user-facing CLI or
# process environment; flip this only in an Agent Bricks release after the managed API is fully deployed.
_USE_MANAGED_RUNTIME_STORE = False

# Agent Bricks deployments use one public prefix for creation and listing.
_DEPLOYMENT_PREFIX = "agent-bricks-"
_AGENTKIT_RUNTIME_STORE_SCHEMA = "databricks_agentkit_runtime"
_MAX_DEPLOYMENT_NAME_LEN = 30  # Databricks Apps name limit
# Instances a deploy pins the app to when `--instances` is not given. Every deploy sends the count, so
# this is the scale an unqualified `agentbricks deploy` settles the app at.
_DEFAULT_INSTANCE_COUNT = 1

# The two env vars the deployed agent reads to enable tracing: a destination (the workspace tracking
# uri) and an experiment (by id). These mirror ``cli.tracing.TRACES_TRACKING_URI_ENV`` /
# ``TRACES_EXPERIMENT_ID_ENV`` (MLflow's own, stable protocol var names); they are defined here rather
# than imported so this module stays free of any ``cli`` import.
TRACES_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
TRACES_EXPERIMENT_ID_ENV = "MLFLOW_EXPERIMENT_ID"


def _validate_deployment_name(name: str, *, check_length: bool = True) -> str:
    """Reject an empty or unsafe deployment name before it reaches a URL / workspace path."""
    if (
        not (name or "").strip()
        or name != name.strip()
        or any(token in name for token in ("/", "\\", ".."))
        or any(character.isspace() for character in name)
    ):
        raise AgentCliError(
            f"Invalid deployment name {name!r}.",
            hint="Use a non-empty name of letters, digits, and hyphens "
            "(no slashes, spaces, or '..').",
        )
    if check_length and len(name) > _MAX_DEPLOYMENT_NAME_LEN:
        raise AgentCliError(
            f"Deployment name {name!r} is too long ({len(name)} > {_MAX_DEPLOYMENT_NAME_LEN}).",
            hint=f"Databricks app names cap at {_MAX_DEPLOYMENT_NAME_LEN} characters, including the "
            f"'{_DEPLOYMENT_PREFIX}' prefix Agent Bricks adds on deploy.",
        )
    return name


class DeploymentName(str):
    """A deployment name that has passed :func:`_validate_deployment_name`; constructing one IS the check.

    Subclasses ``str`` so it drops straight into f-strings, the workspace path, the ``databricks``
    argv, and ``.startswith(_DEPLOYMENT_PREFIX)`` - callers and presenters need no change - while still
    letting the deploy service *require* a validated name in its signatures. A lifecycle verb that
    forgets to validate is then a type error at the call site, not a bug that reaches the workspace -
    the guarantee the per-method ``_validate_deployment_name`` call carried but could not enforce.
    """

    __slots__ = ()

    def __new__(cls, raw: str) -> "DeploymentName":
        return super().__new__(cls, _validate_deployment_name(raw))


def _instance_args(instance_count: int) -> list[str]:
    """Build runtime instance arguments from the Agent Bricks fixed-count option.

    The count is always sent: a deploy pins the app to exactly ``instance_count`` instances (min and
    max alike), so the deployed scale is whatever the current deploy asked for rather than whatever a
    previous deploy happened to leave behind.
    """
    return [
        "--compute-min-instances",
        str(instance_count),
        "--compute-max-instances",
        str(instance_count),
    ]


def _prefixed_name(name: str) -> str:
    """Add the Agent Bricks deployment prefix unless it is already present."""
    return name if name.startswith(_DEPLOYMENT_PREFIX) else f"{_DEPLOYMENT_PREFIX}{name}"


@dataclass(frozen=True)
class MlflowTracingConfig:
    """The MLflow config that binds a deployed agent to its workspace experiment.

    The agent enables tracing when it sees both a destination (the workspace tracking uri) and an
    experiment id; ``env`` renders them as the two env vars wired into app.yaml. (`agentbricks dev` builds
    its own local tracing env instead - see ``cli.tracing.start_local_tracing_server``.)
    """

    experiment_id: str
    tracking_uri: str = "databricks"

    def env(self) -> dict[str, str]:
        """The two ``MLFLOW_*`` env vars to wire into app.yaml, as a name -> value mapping.

        Always both keys, so an empty ``experiment_id`` renders the exact key set a tracing unbind
        has to prune from the manifest.
        """
        return {
            TRACES_TRACKING_URI_ENV: self.tracking_uri,
            TRACES_EXPERIMENT_ID_ENV: self.experiment_id,
        }


def mlflow_tracing_config(experiment_id: str) -> MlflowTracingConfig:
    """The tracing config binding a deployed agent to ``experiment_id``."""
    return MlflowTracingConfig(experiment_id=experiment_id)
