"""Shared `app.yaml` parse/serialize contract for `agentbricks deploy` and `agentbricks dev`.

`AppManifest` centralizes the plumbing both commands need to read and write an app's
`app.yaml`: loading the YAML document, giving callers the raw `env` list, and re-serializing
the document. Each caller keeps its own env policy on top of this - `deploy` upserts/removes
named env entries when reconciling a deployment, while `dev` filters and rewrites the env
list to build a local-only manifest. Neither policy lives here.

Two divergences between the callers are preserved rather than unified:

- Parsing strictness: `deploy` parses leniently via `parse_lenient` (a non-dict top level
  silently becomes an empty document); `dev` parses strictly via `parse` (invalid YAML, a
  non-dict top level, or a non-list `env` all raise `AgentCliError`).
- Env entry shape: `deploy`'s `upsert_env` drops any non-dict entries from `env` before
  upserting; `dev` reads the raw env list via `raw_env` and applies its own filters, which
  preserve non-dict entries untouched.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import yaml

from databricks_agentbricks.errors import AgentCliError

_SCAFFOLD_COMMAND = ["# TODO: set your run command, e.g. ['uvicorn', 'app:app']"]


class AppManifest:
    """An in-memory `app.yaml` document, with parse/serialize helpers shared by deploy and dev."""

    def __init__(self, doc: dict) -> None:
        self._doc = doc

    @classmethod
    def parse(cls, text: str, *, source: pathlib.Path) -> "AppManifest":
        """Strict parse (dev): raise ``AgentCliError`` on invalid YAML, a non-dict top level, or a
        non-list ``env``.
        """
        try:
            doc = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise AgentCliError(f"Could not parse {source}: {exc}") from exc
        if not isinstance(doc, dict):
            raise AgentCliError(f"Invalid {source}: top level must be an object.")
        env = doc.get("env")
        if env is not None and not isinstance(env, list):
            raise AgentCliError(f"Invalid {source}: env must be a list.")
        return cls(doc)

    @classmethod
    def parse_lenient(cls, text: str) -> "AppManifest":
        """Lenient parse (deploy): a non-dict top level silently becomes an empty document."""
        loaded = yaml.safe_load(text)
        return cls(loaded if isinstance(loaded, dict) else {})

    @classmethod
    def scaffold(cls) -> "AppManifest":
        """A fresh manifest for a source dir that has no ``app.yaml`` yet."""
        return cls({"command": _SCAFFOLD_COMMAND, "env": []})

    def raw_env(self) -> list:
        """The document's ``env`` list, or ``[]`` if absent or not a list."""
        raw = self._doc.get("env")
        return raw if isinstance(raw, list) else []

    def upsert_env(self, updates: dict[str, str], removals: Sequence[str] = ()) -> None:
        """deploy's write path: normalize ``env`` to dict entries (dropping non-dicts), upsert each
        name in ``updates`` (setting ``value`` and dropping ``valueFrom`` on an existing entry,
        appending a new one otherwise), then drop any entry named in ``removals``.
        """
        env: list[dict[str, Any]] = [e for e in self.raw_env() if isinstance(e, dict)]
        by_name = {e.get("name"): e for e in env}
        for name, value in updates.items():
            if name in by_name:
                by_name[name]["value"] = value
                by_name[name].pop("valueFrom", None)
            else:
                env.append({"name": name, "value": value})
        if removals:
            drop = set(removals)
            env = [e for e in env if e.get("name") not in drop]
        self._doc["env"] = env

    def set_env(self, entries: list) -> None:
        """dev's write path: replace ``env`` outright with an already-filtered list."""
        self._doc["env"] = entries

    def to_yaml(self) -> str:
        return yaml.safe_dump(self._doc, sort_keys=False)
