#!/usr/bin/env bash
set -euo pipefail

example_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$example_root/../.." && pwd)
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

uv build --wheel "$repository_root" --out-dir "$build_dir"
uv build --wheel "$example_root/shared" --out-dir "$build_dir"

for app_dir in event_log_recovery agent_session_recovery; do
  find "$example_root/$app_dir" -maxdepth 1 -name '*.whl' -delete
  cp "$build_dir"/databricks_ai_bridge-0.21.0-py3-none-any.whl "$example_root/$app_dir/"
  cp "$build_dir"/openai_sdk_agent_shared-0.1.0-py3-none-any.whl "$example_root/$app_dir/"
done
