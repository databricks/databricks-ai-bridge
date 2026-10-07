"""Deployment defaults, rollout settings, and manifest environment values."""

from __future__ import annotations

from dataclasses import dataclass

# TEMPORARY: the Apps build environment currently can't reach the internal pypi proxy, so builds
# time out installing dependencies. Point the build at public PyPI (sanctioned interim workaround)
# until the proxy is reachable from the build sandbox again, then drop this default. pip reads
# PIP_INDEX_URL; uv reads UV_INDEX_URL / UV_DEFAULT_INDEX — set all three to cover both build paths.
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple/"
_PIP_INDEX_ENVS = ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX")
_AGENT_COMPUTE_OUTPUT = ("App compute", "Agent compute")
# Internal rollout switch. Backend selection is intentionally not part of the user-facing CLI or
# process environment; managed provisioning is the release default.
_USE_MANAGED_RUNTIME_STORE = True

_AGENTKIT_RUNTIME_STORE_SCHEMA = "databricks_agentkit_runtime"

# The two env vars the deployed agent reads to enable tracing: a destination (the workspace tracking
# uri) and an experiment (by id). These mirror ``cli.tracing.TRACES_TRACKING_URI_ENV`` /
# ``TRACES_EXPERIMENT_ID_ENV`` (MLflow's own, stable protocol var names); they are defined here rather
# than imported so this module stays free of any ``cli`` import.
TRACES_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
TRACES_EXPERIMENT_ID_ENV = "MLFLOW_EXPERIMENT_ID"


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
