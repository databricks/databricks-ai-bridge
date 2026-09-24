"""Shared data type for a UC-backed experiment's trace tables.

Lives at the package top - NOT under ``databricks_mason.cli`` - so lower-level modules like
``app_resources`` can name it without importing the CLI package. Importing anything from
``databricks_mason.cli`` runs ``cli/__init__`` -> ``cli.app`` -> ``cli.deploy`` -> ``app_resources``,
so a ``cli`` import from ``app_resources`` would be circular (an import-order-dependent failure). The
richer ``MLflowTraceTables`` / ``ResolvedTraceExperiment`` models stay in ``cli.tracing`` (only the CLI
imports those); this holds just the leaf table type that the resource plumbing also needs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceTable:
    """One UC OTEL base table for a trace experiment - its kind ("spans" / "logs" / "annotations" /
    "metrics") and its fully-qualified ``catalog.schema.table`` name."""

    kind: str
    full_name: str
