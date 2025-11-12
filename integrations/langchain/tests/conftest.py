from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_modules_first() -> None:
    tests_dir = Path(__file__).resolve().parent
    langchain_src = tests_dir.parents[2] / "src"
    bridge_src = tests_dir.parents[4] / "src"

    for path in (langchain_src, bridge_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_repo_modules_first()

