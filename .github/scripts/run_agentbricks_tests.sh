#!/usr/bin/env bash
set -euo pipefail

build_wheels() {
  uv build --wheel --out-dir dist ../..
  runtime_packages=()
  runtime_suite_args=()
  # Keep the release runner usable with branches predating the SDK split.
  if test -f src/pyproject.toml; then
    uv build --wheel --out-dir dist src
    runtime_packages=("$PWD"/dist/databricks_agentkit_runtime-*.whl)
    runtime_suite_args=(--runtime-wheel "${runtime_packages[0]}")
  fi
  uv build --wheel --out-dir dist
}

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
    build_wheels
    uv venv .venv-suites
    uv pip install --python .venv-suites/bin/python --no-sources \
      "$(ls dist/databricks_ai_bridge-*.whl)" \
      "${runtime_packages[@]}" \
      "$(ls dist/databricks_agentbricks-*.whl)" --group tests
    # Scaffold projects next to the runner's uv cache: across filesystems uv copies every
    # dependency into each project's venv instead of hardlinking, and that fills /tmp on CI.
    TMPDIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}" \
      .venv-suites/bin/pytest "${@:-tests/suites}" --workspace fake -n auto --dist loadgroup -v --durations=0 \
        --wheel "$(ls "$PWD"/dist/databricks_agentbricks-*.whl)" \
        "${runtime_suite_args[@]}" \
        --bridge-wheel "$(ls "$PWD"/dist/databricks_ai_bridge-*.whl)"
    ;;
  # Release branches cut before tests/suites ran against a fake workspace still carry
  # tests/functional; the release cut runs this script from main against them.
  functional)
    build_wheels
    uv venv .venv-functional
    uv pip install --python .venv-functional/bin/python --no-sources \
      "$(ls dist/databricks_ai_bridge-*.whl)" \
      "${runtime_packages[@]}" \
      "$(ls dist/databricks_agentbricks-*.whl)[loadtest]" --group tests
    .venv-functional/bin/python -c 'import locust'
    AI_BRIDGE_WHEEL="$(ls "$PWD"/dist/databricks_ai_bridge-*.whl)" \
    AGENTBRICKS_WHEEL="$(ls "$PWD"/dist/databricks_agentbricks-*.whl)" \
      .venv-functional/bin/pytest tests/functional/
    ;;
  *)
    echo 'Usage: run_agentbricks_tests.sh {unit {lowest-direct|highest}|framework|suites-fake [tests...]|functional}' >&2
    exit 2
    ;;
esac
