#!/usr/bin/env bash
# Build, deploy and start the dispatcher App. Every Databricks CLI and SDK call here is attributed to the
# connector (DATABRICKS_{CLI,SDK}_UPSTREAM[_VERSION]).
#
#   app/scripts/deploy.sh <target> <profile> <environment id> [--store-secrets [KEY_ENV_VAR]] [-- <bundle args>]
#
# Bundle args (after --) are passed to every bundle command, e.g. --var export_volume=/Volumes/c/s/v/claude.
#
# With --store-secrets, the environment key is read from $KEY_ENV_VAR (default ANTHROPIC_ENVIRONMENT_KEY) and stored
# in the target's secret scope, with a new HMAC key if missing; values are never printed.
set -euo pipefail
target="$1"; profile="$2"; environment_id="$3"; shift 3
store=""; key_env="ANTHROPIC_ENVIRONMENT_KEY"
if [ "${1:-}" = "--store-secrets" ]; then
  store=1; shift
  if [ -n "${1:-}" ] && [ "$1" != "--" ]; then key_env="$1"; shift; fi
fi
[ "${1:-}" = "--" ] && shift
extra=("$@")
here="$(cd "$(dirname "$0")/.." && pwd)"
export DATABRICKS_CLI_UPSTREAM=ClaudeManagedAgents-DatabricksSandbox DATABRICKS_CLI_UPSTREAM_VERSION=0.1.0
export DATABRICKS_SDK_UPSTREAM=ClaudeManagedAgents-DatabricksSandbox DATABRICKS_SDK_UPSTREAM_VERSION=0.1.0
cd "$here"
bundle=(-t "$target" --profile "$profile" --var "environment_id=$environment_id" ${extra[@]+"${extra[@]}"})

scripts/build_wheel.sh
if [ -n "$store" ]; then
  scope="$(databricks bundle validate "${bundle[@]}" -o json | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["resources"]["secret_scopes"]["cma_secrets"]["name"])')"
  # The scope must exist before the App (its secret resources are checked on create).
  databricks bundle deploy "${bundle[@]}" || true
  (cd .. && uv run python app/scripts/put_secrets.py --profile "$profile" --scope "$scope" --key-env "$key_env")
fi
databricks bundle deploy "${bundle[@]}"
databricks bundle run cma_dispatcher "${bundle[@]}"
