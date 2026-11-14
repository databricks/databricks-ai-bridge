#!/usr/bin/env bash
# Build, deploy and start the dispatcher App. Every Databricks CLI and SDK call here is attributed to the
# connector (DATABRICKS_{CLI,SDK}_UPSTREAM[_VERSION]).
#
#   app/scripts/deploy.sh <target> <profile> <environment id> [--store-secrets [KEY_ENV_VAR]] [-- <bundle args>]
#
# Targets (databricks.yml): dev and prod deploy user mode (pass --var user_environment_id=... and, for prod, the
# user_agents and lakebase_* variables); dev-service and prod-service deploy service mode only.
#
# Bundle args (after --) are passed to every bundle command, e.g. --var export_volume=/Volumes/c/s/v/claude.
#
# With --store-secrets, the environment key is read from $KEY_ENV_VAR (default ANTHROPIC_ENVIRONMENT_KEY) and stored
# in the target's secret scope, with a new HMAC key if missing. For a user-mode target, the user environment's key
# ($ANTHROPIC_USER_ENVIRONMENT_KEY) and the App's Anthropic API key ($CMA_ANTHROPIC_API_KEY) are stored too. Values are
# never printed.
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
  # "<scope> <1 if the target is user mode>", from the resolved bundle.
  read -r scope user_mode < <(databricks bundle validate "${bundle[@]}" -o json | python3 -c '
import json, sys
r = json.load(sys.stdin)["resources"]
names = {x["name"] for x in r["apps"]["cma_dispatcher"].get("resources", [])}
print(r["secret_scopes"]["cma_secrets"]["name"], 1 if "anthropic-api-key" in names else 0)')
  user_secrets=()
  if [ "$user_mode" = 1 ]; then
    user_secrets=(--secret api-key=CMA_ANTHROPIC_API_KEY --secret env-u-key=ANTHROPIC_USER_ENVIRONMENT_KEY)
  fi
  # The scope must exist before the App (its secret resources are checked on create).
  databricks bundle deploy "${bundle[@]}" || true
  (cd .. && uv run python app/scripts/put_secrets.py --profile "$profile" --scope "$scope" --key-env "$key_env" \
    ${user_secrets[@]+"${user_secrets[@]}"})
fi
databricks bundle deploy "${bundle[@]}"
databricks bundle run cma_dispatcher "${bundle[@]}"
