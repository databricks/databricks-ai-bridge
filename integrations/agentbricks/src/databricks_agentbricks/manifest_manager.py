"""Reconcile the agentbricks-managed env in an app's ``app.yaml`` manifest.

Render-free collaborator wrapping the ``app.yaml`` env reconcile so a service can drive it without
importing ``click`` or ``render`` (a caller wraps ``reporter.status(...)`` around the call).
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from databricks_agentbricks.app_manifest import AppManifest


class ManifestManager:
    """Upsert/prune the agentbricks-managed env entries in a deployment's ``app.yaml``.

    Stateless: takes no constructor args. The manifest itself (``AppManifest``) owns the parse/scaffold
    /serialize; this only reconciles the env keys Agent Bricks manages against the desired state.
    """

    def upsert_env(
        self,
        source: pathlib.Path,
        updates: dict[str, str],
        removals: Sequence[str] = (),
    ) -> bool:
        """Reconcile env entries in <source>/app.yaml: upsert ``updates``, drop any named in ``removals``.

        Returns True if it scaffolded a new file. ``removals`` lets an unbind clear stale agentbricks-managed env
        (e.g. the ``MLFLOW_*`` keys when tracing is unbound) so the manifest stops pointing the deployed
        runtime at a resource whose grant has just been pruned; without it, the upsert-only merge would
        leave the stale entry behind. ``updates`` and ``removals`` are expected to be disjoint.
        """
        app_yaml = source / "app.yaml"
        if app_yaml.exists():
            manifest = AppManifest.parse_lenient(app_yaml.read_text())
            scaffolded = False
        else:
            manifest = AppManifest.scaffold()
            scaffolded = True

        manifest.upsert_env(updates, removals)
        app_yaml.write_text(manifest.to_yaml())
        return scaffolded
