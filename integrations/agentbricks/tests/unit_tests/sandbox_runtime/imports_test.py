"""Cold-import read-ahead stays bounded and never changes normal agent execution."""

import json
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest

from databricks_agentkit.sandbox.adapters import imports


@pytest.fixture
def installed(tmp_path, monkeypatch):
    prefix = tmp_path / "venv"
    library = prefix / "lib/python3.12/site-packages"
    library.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(imports, "_prefetched", False)
    monkeypatch.delenv("AGENTBRICKS_IMPORT_PREFETCH", raising=False)
    state = tmp_path / "session"
    state.mkdir()
    opened, read_started = [], threading.Event()
    original = Path.open

    def track(path, *args, **kwargs):
        if args and args[0] == "rb":
            opened.append(path)
            read_started.set()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track)
    return prefix, library, state, opened, read_started


def test_new_worker_skips_inventory_and_learns_for_next_process(installed, monkeypatch):
    prefix, library, state, opened, read_started = installed
    code = library / "example.py"
    code.write_text("raise AssertionError('read-ahead must not execute code')")
    record = library / "example.dist-info/RECORD"
    record.parent.mkdir()
    native = library / "native.so"
    native.write_bytes(b"native extension content")
    record.write_text("example.py,,\nnative.so,,\n../outside.py,,\nsettings.json,,\n")
    module = ModuleType("imported_example")
    module.__file__ = str(code)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    timings = {}
    with imports.warm_imports(state, timings):
        assert not opened
    assert not opened
    manifest = json.loads((state / "import-files.json").read_text())
    assert manifest == {"prefix": str(prefix), "files": [str(code.relative_to(prefix))]}
    assert timings["import_cache"] >= 0
    opened.clear()
    read_started.clear()
    # The same worker does no repeated read-ahead.
    with imports.warm_imports(state, {}):
        pass
    assert opened == []
    # A replacement worker can use the learned list without RECORD discovery.
    monkeypatch.setattr(imports, "_prefetched", False)
    with imports.warm_imports(state, {}):
        assert read_started.wait(2)
    assert code in opened


def test_cache_filters_escape_paths_symlinks_and_non_code(installed):
    prefix, library, state, opened, read_started = installed
    code = library / "example.py"
    code.write_text("pass")
    outside = prefix.parent / "outside.py"
    outside.write_text("pass")
    link = library / "link.py"
    link.symlink_to(outside)
    settings = library / "settings.json"
    settings.write_text("{}")
    (state / "import-files.json").write_text(
        json.dumps(
            {
                "prefix": str(prefix),
                "files": [
                    str(code.relative_to(prefix)),
                    "../outside.py",
                    str(outside),
                    str(link.relative_to(prefix)),
                    str(settings.relative_to(prefix)),
                    9,
                ],
            }
        )
    )
    with imports.warm_imports(state, {}):
        assert read_started.wait(2)
    assert code in opened
    assert outside not in opened and link not in opened and settings not in opened


@pytest.mark.parametrize("contents", ["bad JSON", '{"prefix": [], "files": 1}'])
def test_invalid_cache_preserves_agent_failure(installed, contents):
    _, _, state, _, _ = installed
    (state / "import-files.json").write_text(contents)
    with pytest.raises(RuntimeError, match="actual agent failure"):
        with imports.warm_imports(state, {}):
            raise RuntimeError("actual agent failure")


def test_read_ahead_has_a_byte_budget_and_an_opt_out(installed, monkeypatch):
    prefix, library, state, opened, read_started = installed
    code = library / "oversized.py"
    code.write_bytes(b"x" * 20)
    (state / "import-files.json").write_text(
        json.dumps(
            {
                "prefix": str(prefix),
                "files": [str(code.relative_to(prefix))],
            }
        )
    )
    monkeypatch.setattr(imports, "MAX_BYTES", 10)
    with imports.warm_imports(state, {}):
        pass
    assert code not in opened
    monkeypatch.setattr(imports, "_prefetched", False)
    monkeypatch.setenv("AGENTBRICKS_IMPORT_PREFETCH", "0")
    monkeypatch.setattr(imports, "_paths", lambda *args: pytest.fail("opt out ignored"))
    with imports.warm_imports(state, {}):
        pass


@pytest.mark.parametrize("failure", ["manifest", "executor"])
def test_read_ahead_setup_failure_still_runs_agent_once(installed, monkeypatch, failure):
    _, _, state, _, _ = installed

    def fail(*args, **kwargs):
        raise RuntimeError("cache setup unavailable")

    monkeypatch.setattr(imports, "_paths" if failure == "manifest" else "ThreadPoolExecutor", fail)
    calls = []
    with imports.warm_imports(state, {}):
        calls.append("normal imports")
    assert calls == ["normal imports"]


def test_failed_imports_preserve_learned_manifest(installed):
    prefix, _, state, _, _ = installed
    manifest = state / "import-files.json"
    saved = json.dumps({"prefix": str(prefix), "files": []})
    manifest.write_text(saved)
    with pytest.raises(RuntimeError, match="agent import failed"):
        with imports.warm_imports(state, {}):
            raise RuntimeError("agent import failed")
    assert manifest.read_text() == saved


def test_cached_bytecode_symlink_cannot_escape_prefix(installed):
    prefix, library, state, opened, _ = installed
    code = library / "example.py"
    code.write_text("pass")
    from importlib.util import cache_from_source

    cached = Path(cache_from_source(str(code)))
    cached.parent.mkdir()
    outside = prefix.parent / "outside.pyc"
    outside.write_bytes(b"outside")
    cached.symlink_to(outside)
    (state / "import-files.json").write_text(
        json.dumps(
            {
                "prefix": str(prefix),
                "files": [str(code.relative_to(prefix))],
            }
        )
    )
    with imports.warm_imports(state, {}):
        pass
    assert outside not in opened and cached not in opened
