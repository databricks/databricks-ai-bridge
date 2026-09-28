"""Shared data types for a UC-backed experiment's trace tables.

Lives at the package top - NOT under ``databricks_agentbricks.cli`` - so lower-level modules like
``app_resources`` can name these without importing the CLI package. Importing anything from
``databricks_agentbricks.cli`` runs ``cli/__init__`` -> ``cli.app`` -> ``cli.deploy`` -> ``app_resources``,
so a ``cli`` import from ``app_resources`` would be circular (an import-order-dependent failure). The
richer ``MLflowTraceTables`` / ``ResolvedTraceExperiment`` models stay in ``cli.tracing`` (only the CLI
imports those); this holds just the leaf types the resource plumbing also needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TraceTableKind(str, Enum):
    """The kind of UC OTEL base table backing an experiment's traces.

    A ``str`` mixin so a member compares equal to its literal value; use ``.value`` when building
    strings (e.g. the app-resource name) so the rendering is the bare kind, not ``TraceTableKind.X``.
    """

    SPANS = "spans"
    LOGS = "logs"
    ANNOTATIONS = "annotations"
    METRICS = "metrics"


@dataclass(frozen=True)
class TraceTable:
    """One UC OTEL base table for a trace experiment - its ``kind`` and its fully-qualified
    ``catalog.schema.table`` name."""

    kind: TraceTableKind
    full_name: str
