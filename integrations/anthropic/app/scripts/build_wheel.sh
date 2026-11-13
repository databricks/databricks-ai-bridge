#!/usr/bin/env bash
# Build the connector wheel (from the package next to app/) and point requirements.txt's wheel line at it.
# Once databricks-anthropic is on PyPI, replace that line with `databricks-anthropic[mcp]==<version>` instead.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf "$here/wheels" && mkdir -p "$here/wheels"
uv build --quiet --wheel --out-dir "$here/wheels" "$here/.."
wheel="$(cd "$here/wheels" && ls databricks_anthropic-*.whl)"
sed -i.bak -E "s#^\./wheels/databricks_anthropic-.*\.whl\$#./wheels/${wheel}#" "$here/requirements.txt"
rm -f "$here/requirements.txt.bak"
echo "built wheels/${wheel}"
