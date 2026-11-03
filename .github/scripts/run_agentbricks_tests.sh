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
    uv run --resolution highest --exact --group tests --extra langgraph --extra openai --extra maf pytest tests/unit_tests
    generated_templates=$(mktemp -d)
    trap 'rm -rf "$generated_templates"' EXIT
    for framework in openai maf; do
      AGENTBRICKS_DISABLE_TELEMETRY=1 AGENTBRICKS_DISABLE_AGENT_HINT=1 \
        uv run --no-sync agentbricks init --framework "$framework" "$generated_templates/agent-$framework"
      PYTHONPATH="$generated_templates/agent-$framework" \
        uv run --no-sync pytest "$generated_templates/agent-$framework/tests"
    done
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
