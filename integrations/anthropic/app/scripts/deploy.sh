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
# Recommended for user mode: -- --var user_memory=1 (each user gets their own Databricks memory store; needs the
# Agent Memory Sessions preview and the memory tools on the agent). Off by default.
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

# The bundle rewrites the secret scope's ACL (MANAGE for the manager only) before it updates the App, and the App's
# READ comes back only with a successful App update. If the deploy fails in between, grant the App's READ again so
# the running App can still read its secrets when it restarts.
restore_app_read() {
  local scope app sp
  read -r scope app < <(databricks bundle validate "${bundle[@]}" -o json | python3 -c '
import json, sys
r = json.load(sys.stdin)["resources"]
print(r["secret_scopes"]["cma_secrets"]["name"], r["apps"]["cma_dispatcher"]["name"])') || return 0
  sp=$(databricks apps get "$app" --profile "$profile" -o json 2>/dev/null \
    | python3 -c 'import json, sys; print(json.load(sys.stdin).get("service_principal_client_id") or "")') || sp=""
  if [ -n "$sp" ]; then
    databricks secrets put-acl "$scope" "$sp" READ --profile "$profile"
    echo "deploy failed: granted the App's service principal READ on $scope again" >&2
  fi
}

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
if ! databricks bundle deploy "${bundle[@]}"; then
  restore_app_read
  exit 1
fi
databricks bundle run cma_dispatcher "${bundle[@]}"
