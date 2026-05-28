"""Analytical functions derived from the cached query layer.

Every public function here is a ``compute_*`` that takes no DB
connection of its own — instead it composes the cached ``get_*``
functions from :mod:`._components.data_access`. The split keeps the
math testable against synthetic DataFrames without spinning up DuckDB.

Functions in this module are also cached via ``@st.cache_data`` so
the System page and the Overview can both call them without paying
the recomputation cost twice.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .data_access import (
    NOTABLE_FINDINGS,
    SYSTEMS,
    get_active_status,
    get_daily_availability,
    get_daily_kpis,
    get_identity,
    get_threshold_events,
)


# ─── Per-system status (the Overview headline) ────────────────────────
@st.cache_data(ttl=3600)
def compute_system_status(window_days: int = 30) -> pd.DataFrame:
    """Per-system snapshot for the Overview status table.

    Reads the daily KPI table and the threshold-event log, then for
    each system computes:

    * ``last_seen`` and ``is_active`` — retirement classification
    * ``rte_pct``, ``mean_dt_c`` — last-`window` aggregates relative
      to that *system's own* last sample (not the fleet's), so we
      don't show "—" for racks that retired before the fleet-wide
      cutoff
    * ``warning_events``, ``critical_events`` — last-`window`
      severity counts
    * ``has_notable_finding`` — boolean override from
      :data:`NOTABLE_FINDINGS`
    * ``status`` — operator pill, one of:
          ``healthy``  → no warnings, no notable finding, OK KPIs
          ``watch``    → any warnings, ΔT > 5 °C, or notable finding
          ``critical`` → any critical event
          ``retired``  → no telemetry within the gap window
    """
    kpis = get_daily_kpis()
    events = get_threshold_events()
    active = get_active_status().set_index("system_id")

    rows: list[dict] = []
    for sid in SYSTEMS:
        a = active.loc[sid]
        sys_max = a["last_seen"]
        cutoff_k = sys_max - pd.Timedelta(days=window_days)
        recent_k = kpis[
            (kpis["system_id"] == sid) & (kpis["date"] > cutoff_k)
        ]
        rte = recent_k["rte"].median() * 100 if not recent_k.empty else np.nan
        dt = recent_k["mean_dt_c"].mean() if not recent_k.empty else np.nan

        recent_e = events[
            (events["system_id"] == sid)
            & (events["start"] > cutoff_k)
        ]
        crit = int((recent_e["severity"] == "critical").sum())
        warn = int((recent_e["severity"] == "warning").sum())

        finding = NOTABLE_FINDINGS.get(sid)

        if not a["is_active"]:
            pill = "retired"
        elif crit > 0:
            pill = "critical"
        elif warn > 0 or (pd.notna(dt) and dt > 5.0) or finding is not None:
            pill = "watch"
        else:
            pill = "healthy"

        rows.append({
            "system_id":            sid,
            "last_seen":            a["last_seen"],
            "days_since_seen":      int(a["days_since_seen"]),
            "is_active":            bool(a["is_active"]),
            "rte_pct":              rte,
            "mean_dt_c":            dt,
            "warning_events":       warn,
            "critical_events":      crit,
            "has_notable_finding":  finding is not None,
            "notable_finding":      finding,
            "status":               pill,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def compute_availability(window_days: int = 30) -> pd.DataFrame:
    """Per-system median availability over the recent window.

    Reads from :func:`.data_access.get_daily_availability`
    (DST-capped, interpolation-discounted). Window is relative to each
    system's own last sample, so retired racks report their final-month
    aggregates rather than showing NaN against a fleet-wide cutoff.
    """
    avail = get_daily_availability()
    active = get_active_status().set_index("system_id")
    rows: list[dict] = []
    for sid in SYSTEMS:
        sys_max = active.loc[sid, "last_seen"]
        cutoff = sys_max - pd.Timedelta(days=window_days)
        recent = avail[(avail["system_id"] == sid) & (avail["date"] > cutoff)]
        med = float(recent["availability_pct"].median()) if not recent.empty else float("nan")
        rows.append({"system_id": sid, "availability_pct": med})
    return pd.DataFrame(rows)


