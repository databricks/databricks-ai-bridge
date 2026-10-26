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
    uv run --resolution highest --exact --group tests --extra langgraph --extra openai pytest tests/unit_tests
    PYTHONPATH=src/databricks_agentbricks/templates/agent-openai \
      uv run --no-sync pytest src/databricks_agentbricks/templates/agent-openai/tests
    ;;
  suites-fake)
    # `agentbricks dev` runs the app through `databricks apps run-local`, so the CLI must be on PATH.
    # Any further arguments select the tests to run (default: every suite).
    shift
    uv build --wheel --out-dir dist ../..
    uv build --wheel --out-dir dist
    uv venv .venv-suites
    uv pip install --python .venv-suites/bin/python --no-sources \
      "$(ls dist/databricks_ai_bridge-*.whl)" \
      "$(ls dist/databricks_agentbricks-*.whl)" --group tests
    # Scaffold projects next to the runner's uv cache: across filesystems uv copies every
    # dependency into each project's venv instead of hardlinking, and that fills /tmp on CI.
    TMPDIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}" \
      .venv-suites/bin/pytest "${@:-tests/suites}" --workspace fake -n auto --dist loadgroup \
        --wheel "$(ls "$PWD"/dist/databricks_agentbricks-*.whl)" \
        --bridge-wheel "$(ls "$PWD"/dist/databricks_ai_bridge-*.whl)"
    ;;
  # Release branches cut before tests/suites ran against a fake workspace still carry
  # tests/functional; the release cut runs this script from main against them.
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
    echo 'Usage: run_agentbricks_tests.sh {unit {lowest-direct|highest}|framework|suites-fake [tests...]|functional}' >&2
    exit 2
    ;;
esac
