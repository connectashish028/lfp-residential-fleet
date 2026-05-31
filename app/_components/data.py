"""Backwards-compatible facade for the data layer.

This module used to hold both the cached DuckDB queries and the
analytical helpers; it has been split into two for clarity:

* :mod:`._components.data_access` — cached ``get_*`` queries that
  return DataFrames straight from the DuckDB views.
* :mod:`._components.analytics`   — cached ``compute_*`` functions
  that derive analytical values (status pills, availability) by
  composing the queries above.

Pages still import via ``from _components import data`` and call
``data.get_X()`` or ``data.compute_X()`` — that contract is preserved
by re-exporting the public surface from both sub-modules here.

New code should prefer importing directly from the sub-module, e.g.
``from _components.data_access import get_daily_kpis`` — it's
clearer where the function lives.
"""
from __future__ import annotations

from .analytics import (
    compute_availability,
    compute_degradation_summary,
    compute_system_status,
)
from .data_access import (
    NOTABLE_FINDINGS,
    RETIREMENT_GAP_DAYS,
    SYSTEMS,
    get_active_status,
    get_daily_availability,
    get_daily_kpis,
    get_daily_soc_spread,
    get_daily_voltage_spread,
    get_data_window,
    get_degradation_modes,
    get_degradation_summary,
    get_identity,
    get_telemetry,
    get_telemetry_bounds,
    get_threshold_events,
)

__all__ = [
    # constants
    "SYSTEMS", "RETIREMENT_GAP_DAYS", "NOTABLE_FINDINGS",
    # data access
    "get_identity", "get_daily_kpis", "get_threshold_events",
    "get_telemetry", "get_telemetry_bounds", "get_data_window",
    "get_active_status", "get_daily_availability",
    "get_daily_soc_spread", "get_daily_voltage_spread",
    "get_degradation_modes", "get_degradation_summary",
    # analytics
    "compute_system_status",
    "compute_availability",
    "compute_degradation_summary",
]
