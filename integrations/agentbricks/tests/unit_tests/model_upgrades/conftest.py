"""Fixtures for the model-upgrade optimizer tests (migrated from Smart Model Upgrades).

Keeps the suite hermetic:

- No workspace calls. ``optimization._client()`` returns a stand-in whose serving-endpoint lookup
  fails, so model names resolve through the ``system.ai.<name>`` fallback; tests that touch a model
  service patch the ``model_services`` function they exercise.
- No pricing-catalog calls. ``_mlflow_model_cost`` normally reaches MLflow's pricing catalog, whose
  result varies with network access and catalog contents; it's stubbed with a fixed rate for
  priceable names (and None otherwise, mirroring a catalog miss). Tests that assert on a specific
  cost patch ``_mlflow_model_cost`` / ``_estimate_cost_usd`` directly; ``@pytest.mark.real_pricing``
  opts out of the stub.

The optimizer needs the ``upgrade`` extra (gepa, full mlflow); without it this directory is skipped.
"""

from __future__ import annotations

from unittest import mock

import pytest

try:
    import gepa  # noqa: F401
    import mlflow.genai  # noqa: F401
except ImportError:
    collect_ignore_glob = ["*_test.py"]

# Nominal USD per token for priceable models. Arbitrary but fixed so tests are deterministic.
_STUB_INPUT_USD_PER_TOKEN = 1e-6
_STUB_OUTPUT_USD_PER_TOKEN = 3e-6
_OPT = "databricks_agentkit.model_upgrades.optimization"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_pricing: exercise the real _mlflow_model_cost wrapper (no stub)",
    )


class _Mocker:
    """The slice of pytest-mock's ``mocker`` these tests use: ``patch`` and ``Mock``, auto-undone."""

    Mock = mock.Mock
    MagicMock = mock.MagicMock

    def __init__(self):
        self._patches: list = []

    def patch(self, target, *args, **kwargs):
        patcher = mock.patch(target, *args, **kwargs)
        self._patches.append(patcher)
        return patcher.start()

    def stopall(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        self._patches.clear()


@pytest.fixture
def mocker():
    m = _Mocker()
    yield m
    m.stopall()


class _NoWorkspace:
    """Stands in for the WorkspaceClient: any serving-endpoint lookup fails like a missing endpoint."""

    class serving_endpoints:  # noqa: N801 - mirrors the SDK attribute
        @staticmethod
        def get(name):
            raise LookupError(f"no serving endpoint {name!r} in tests")


@pytest.fixture(autouse=True)
def _no_workspace(mocker):
    mocker.patch(f"{_OPT}._client", return_value=_NoWorkspace())
    mocker.patch(f"{_OPT}._model_info_cache", {})


@pytest.fixture(autouse=True)
def _stub_mlflow_model_cost(request, mocker):
    if request.node.get_closest_marker("real_pricing"):
        return

    # Prefixes MLflow's catalog can price: gateway aliases (databricks-*) plus the provider-native
    # resolved names FMAPI echoes (gpt-*, claude-*, gemini-*, llama-*).
    priceable = ("databricks-", "gpt-", "claude-", "gemini-", "llama-")

    def _fake_cost(model, in_tokens, out_tokens):
        if not isinstance(model, str) or not model.startswith(priceable):
            return None
        return in_tokens * _STUB_INPUT_USD_PER_TOKEN + out_tokens * _STUB_OUTPUT_USD_PER_TOKEN

    mocker.patch(f"{_OPT}._mlflow_model_cost", side_effect=_fake_cost)
