"""Shared `app.yaml` parse/serialize contract for `agentbricks deploy` and `agentbricks dev`.

`AppManifest` centralizes the plumbing both commands need to read and write an app's
`app.yaml`: loading the YAML document, giving callers the raw `env` list, and re-serializing
the document. Each caller keeps its own env policy on top of this - `deploy` upserts/removes
named env entries when reconciling a deployment, while `dev` filters and rewrites the env
list to build a local-only manifest. `dev`'s policy lives with `dev`; `deploy`'s lives here,
as `AppManifest.upsert_env_file`, because it is the same merge `AppManifest.upsert_env`
implements, one level up (read the file, apply, write back) - and splitting the two halves across
modules only hid that. `AppManifest.require_deployable` is deploy's preflight: deploy never writes a
new `app.yaml`, so it refuses a source without a usable one before touching the workspace.

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
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

from databricks_agentbricks.errors import AgentCliError

# Scaffold placeholder commands; deploy refuses them because such an app.yaml can never start.
_PLACEHOLDER_MARKERS = ("# todo: set your run command", "replace_with")


def is_placeholder_command(command: object) -> bool:
    """Whether an app.yaml ``command`` (a string or list) is still a scaffold/template placeholder."""
    parts = command if isinstance(command, list) else [command]
    rendered = " ".join(str(part) for part in parts if part is not None).casefold()
    return any(marker in rendered for marker in _PLACEHOLDER_MARKERS)


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
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise AgentCliError(f"Could not parse {source}: {exc}") from exc
        if doc is None:  # An empty or comment-only file is an empty document, not a falsey value.
            doc = {}
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
    def require_deployable(cls, source: pathlib.Path) -> None:
        """deploy's preflight: raise ``AgentCliError`` unless <source>/app.yaml exists, parses
        strictly, and has no placeholder command. Paths in the error are absolute.
        """
        directory = source.resolve()
        app_yaml = directory / "app.yaml"
        if not app_yaml.is_file():
            if (directory / "agent.toml").is_file():
                raise AgentCliError(
                    f"No app.yaml in {directory}.",
                    hint="Add an app.yaml with your start command, or re-create the project with "
                    "`agentbricks init`.",
                )
            raise AgentCliError(
                f"No Agent Bricks project in {directory} (no agent.toml or app.yaml).",
                hint="Run from your project directory, pass --source <dir>, or create one with "
                "`agentbricks init`.",
            )
        command = cls.parse(app_yaml.read_text(), source=app_yaml)._doc.get("command")
        if is_placeholder_command(command):
            raise AgentCliError(
                f"{app_yaml} has a placeholder command.",
                hint="Set a real `command:` in it, or create a project with `agentbricks init`.",
            )

    def raw_env(self) -> list:
        """The document's ``env`` list, or ``[]`` if absent or not a list."""
        raw = self._doc.get("env")
        return raw if isinstance(raw, list) else []

    def upsert_env(
        self,
        updates: Mapping[str, str],
        removals: Sequence[str] = (),
        *,
        prune_prefixes: Sequence[str] = (),
    ) -> None:
        """deploy's write path: normalize ``env`` to dict entries (dropping non-dicts), upsert each
        name in ``updates`` (setting ``value`` and dropping ``valueFrom`` on an existing entry,
        appending a new one otherwise), then drop any entry named in ``removals``.
        """
        env: list[dict[str, Any]] = [e for e in self.raw_env() if isinstance(e, dict)]
        if prune_prefixes:
            env = [
                entry
                for entry in env
                if not str(entry.get("name", "")).startswith(tuple(prune_prefixes))
                or entry.get("name") in updates
            ]
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
        """Serialize the document back to YAML, preserving key order (``app.yaml`` is user-edited)."""
        return yaml.safe_dump(self._doc, sort_keys=False)

    @classmethod
    def upsert_env_file(
        cls,
        source: pathlib.Path,
        updates: Mapping[str, str],
        removals: Sequence[str] = (),
        *,
        prune_prefixes: Sequence[str] = (),
    ) -> None:
        """Reconcile env entries in <source>/app.yaml: upsert ``updates``, drop any named in ``removals``.

        The file-level half of ``deploy``'s env policy - read, apply, write back - next to the
        document-level half (:meth:`AppManifest.upsert_env`) it drives, so the manifest contract lives in
        one module. It never creates app.yaml; deploy checks it exists up front
        (:meth:`AppManifest.require_deployable`).

        ``removals`` lets an unbind clear stale agentbricks-managed env (e.g. the ``MLFLOW_*`` keys when
        tracing is unbound) so the manifest stops pointing the deployed runtime at a resource that is no
        longer configured; without it, the upsert-only merge would leave the stale entry behind.
        ``updates`` and ``removals`` are expected to be disjoint.
        """
        app_yaml = source / "app.yaml"
        if not app_yaml.is_file():
            # Guard: require_deployable ran first, so app.yaml only goes missing mid-deploy.
            raise AgentCliError(f"No app.yaml in {source.resolve()}.")
        manifest = cls.parse_lenient(app_yaml.read_text())
        manifest.upsert_env(updates, removals, prune_prefixes=prune_prefixes)
        app_yaml.write_text(manifest.to_yaml())
