#!/usr/bin/env bash
set -euo pipefail

# Run from the Agent Bricks package directory in the checkout being tested.
case "${1:-}" in
  unit)
    resolution="${2:?Provide a resolution: lowest-direct or highest}"
    case "$resolution" in
      lowest-direct|highest) ;;
      *) echo "Unsupported resolution: $resolution" >&2; exit 2 ;;
    esac
    uv run --resolution "$resolution" --exact --group tests pytest tests/unit_tests
    ;;
  framework)
    uv run --resolution highest --exact --group tests --extra langgraph pytest tests/unit_tests
    ;;
  functional)
    uv build --wheel --out-dir dist ../..
    uv build --wheel --out-dir dist
    uv venv .venv-functional
    uv pip install --python .venv-functional/bin/python --no-sources \
      "$(ls dist/databricks_ai_bridge-*.whl)" \
      "$(ls dist/databricks_agentbricks-*.whl)" --group tests
    AI_BRIDGE_WHEEL="$(ls "$PWD"/dist/databricks_ai_bridge-*.whl)" \
    AGENTBRICKS_WHEEL="$(ls "$PWD"/dist/databricks_agentbricks-*.whl)" \
      .venv-functional/bin/pytest tests/functional/
    ;;
  *)
    echo 'Usage: run_agentbricks_tests.sh {unit {lowest-direct|highest}|framework|functional}' >&2
    exit 2
    ;;
esac
