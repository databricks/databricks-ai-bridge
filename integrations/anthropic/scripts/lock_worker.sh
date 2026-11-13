#!/usr/bin/env bash
# Regenerate the hash-locked requirements the in-sandbox worker venv installs (--require-hashes).
# Inputs are the exact pins in launcher.py (WORKER_PACKAGES, MCP_PACKAGES); run after changing them.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$here/src/databricks_anthropic/sandbox"
pins="$(cd "$here" && uv run --quiet python -c '
from databricks_anthropic.sandbox.launcher import MCP_PACKAGES, WORKER_PACKAGES
print("\n".join(WORKER_PACKAGES)); print("#mcp"); print("\n".join(MCP_PACKAGES))')"
base="$(printf '%s\n' "$pins" | sed '/^#mcp$/,$d')"
mcp="$(printf '%s\n' "$pins" | sed -n '/^#mcp$/,$p' | sed 1d)"
compile() {  # universal: one lock for every platform and Python >= 3.10 the sandbox image may have
  uv pip compile --quiet --universal --python-version 3.10 --generate-hashes --no-header \
    --custom-compile-command "scripts/lock_worker.sh" - -o "$1"
}
printf '%s\n' "$base" | compile "$out/worker-requirements.txt"
printf '%s\n%s\n' "$base" "$mcp" | compile "$out/worker-requirements-mcp.txt"
echo "wrote $out/worker-requirements.txt $out/worker-requirements-mcp.txt"
