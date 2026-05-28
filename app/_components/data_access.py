"""Cached DuckDB query layer.

Every public function here is a ``get_*`` that returns a pandas
DataFrame straight from a DuckDB view. Functions are decorated with
``@st.cache_data`` so navigation doesn't re-hit the database.

This module is the *only* place the dashboard talks to DuckDB. Pages
and analytics modules must not import ``bess_fleet.db`` directly —
go through here so the cache layer can do its job.

For analytical functions that derive values from these queries
(status pills, availability), see :mod:`._components.analytics`.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from bess_fleet.db import connect

# Visible systems for today's demo — scoped down to ID16 + ID17 to
# keep the System page narrative tight. The full fleet
# (ID14, ID16, ID17, ID18, ID19, ID20) is in the parquets; expanding
# this list re-enables the others on the dashboard without any other
# code change.
SYSTEMS: list[str] = ["ID16", "ID17"]

# A system is "retired" when its last telemetry sample is older than
# the fleet's most-recent sample by more than this gap. Set high enough
# to absorb dataset-ingest seams (DuckDB views can lag a few minutes)
# but low enough to catch racks that genuinely stopped reporting weeks
# or months back.
RETIREMENT_GAP_DAYS: int = 60

# Hand-curated overrides — racks whose status pill should escalate to
# at least "watch" because an engineer has investigated and flagged a
# root cause. Empty by default; add entries here when a curated finding
# should override the event-count-based pill.
NOTABLE_FINDINGS: dict[str, str] = {}


# ─── Reference tables ──────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_identity() -> pd.DataFrame:
    """Per-system metadata (capacity, voltage, cells, install date)."""
    with connect() as con:
        return con.sql("SELECT * FROM identity ORDER BY system_id").df()


@st.cache_data(ttl=3600)
def get_daily_kpis() -> pd.DataFrame:
    """One row per (system, day). Pre-cleaned dates."""
    with connect() as con:
        df = con.sql("SELECT * FROM daily_kpis ORDER BY system_id, date").df()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def get_threshold_events() -> pd.DataFrame:
    """All rule-based events, sorted by start time (newest first)."""
    with connect() as con:
        df = con.sql('SELECT * FROM threshold_events ORDER BY "start" DESC').df()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


# ─── Telemetry — windowed, capped so we bound entries ─────────────────
@st.cache_data(ttl=3600, max_entries=10, show_spinner="loading telemetry…")
def get_telemetry(system_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """1-min telemetry for one system within a date window."""
    with connect() as con:
        df = con.sql(f"""
            SELECT * FROM telemetry_1min_clean
            WHERE system_id = '{system_id}'
              AND timestamp >= TIMESTAMP '{start.isoformat()}'
              AND timestamp <  TIMESTAMP '{end.isoformat()}'
            ORDER BY timestamp
        """).df()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(ttl=3600)
def get_telemetry_bounds() -> pd.DataFrame:
    """Per-system min / max timestamp."""
    with connect() as con:
        df = con.sql("""
            SELECT system_id,
                   MIN(timestamp) AS min_ts,
                   MAX(timestamp) AS max_ts
            FROM telemetry_1min_clean
            GROUP BY system_id ORDER BY system_id
        """).df()
    df["min_ts"] = pd.to_datetime(df["min_ts"], utc=True).dt.tz_localize(None)
    df["max_ts"] = pd.to_datetime(df["max_ts"], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(ttl=3600)
def get_data_window() -> dict[str, pd.Timestamp]:
    """Fleet-wide telemetry window — used for the Historical-Replay
    banner and per-system retirement detection."""
    b = get_telemetry_bounds()
    return {
        "fleet_min": pd.Timestamp(b["min_ts"].min()),
        "fleet_max": pd.Timestamp(b["max_ts"].max()),
    }


@st.cache_data(ttl=3600)
def get_active_status() -> pd.DataFrame:
    """Per-system active / retired classification.

    A system is retired if its last telemetry sample is more than
    :data:`RETIREMENT_GAP_DAYS` older than the fleet's most-recent
    sample. Returns one row per system with ``last_seen``,
    ``days_since_seen``, ``is_active``.
    """
    bounds = get_telemetry_bounds()
    fleet_max = bounds["max_ts"].max()
    df = bounds.copy()
    df["last_seen"] = df["max_ts"]
    df["days_since_seen"] = (fleet_max - df["max_ts"]).dt.days
    df["is_active"] = df["days_since_seen"] <= RETIREMENT_GAP_DAYS
    return df[["system_id", "last_seen", "days_since_seen", "is_active"]]


@st.cache_data(ttl=3600)
def get_daily_availability() -> pd.DataFrame:
    """Per-(system, date) availability series — single source of truth.

    Definition::

        availability_pct  = min(100, real_minutes / 1440 × 100)
        real_minutes      = n_samples × (1 − mean(interpolated_frac))

    Two corrections vs the raw ``coverage_pct`` in ``daily_kpis``:

    1. **DST cap.** Fall-back DST transition days contain 25 hours
       (1500 1-minute rows) of local-calendar timestamps. The raw
       ratio leaks above 100 %; we cap at 100 % so the metric stays
       interpretable.
    2. **Interpolation discount.** ``interpolated_frac`` in the
       cleaned view marks 1-min rows whose underlying 1-second samples
       were reconstructed by Figgener's gap-filling. A row with
       ``interpolated_frac = 1.0`` is fully synthetic — we don't have
       data, we have neighbours' data — so we down-weight it. Stricter
       than naive ``COUNT(*)/1440`` and more honest about real data
       presence.

    Columns: ``system_id``, ``date``, ``availability_pct``.
    """
    with connect() as con:
        df = con.sql("""
            SELECT
                system_id,
                date_trunc('day', timestamp)::DATE AS date,
                COUNT(*)                                  AS n_samples,
                AVG(COALESCE(interpolated_frac, 0))       AS mean_interp
            FROM telemetry_1min_clean
            GROUP BY system_id, date
            ORDER BY system_id, date
        """).df()
    df["date"] = pd.to_datetime(df["date"])
    real_minutes = df["n_samples"] * (1.0 - df["mean_interp"].clip(0, 1))
    df["availability_pct"] = (real_minutes / 1440.0 * 100.0).clip(upper=100.0)
    return df[["system_id", "date", "availability_pct"]]


@st.cache_data(ttl=3600)
def get_daily_soc_spread(system_id: str) -> pd.DataFrame:
    """Daily SoC range — single-pack analogue of the container-
    level SoC-imbalance chart. Returns ``date``, ``spread`` (= max−min
    SoC % per calendar day), ``soc_max``, ``soc_min``.

    With one pack per system we can't observe inter-cell imbalance,
    so the daily range is the closest physical analogue: a wide
    daily SoC swing = deep cycling, narrow swing = light cycling.
    Days with insufficient data return NaN spread.
    """
    with connect() as con:
        df = con.sql(f"""
            SELECT date_trunc('day', timestamp)::DATE AS date,
                   MAX(soc_pct) - MIN(soc_pct)     AS spread,
                   MAX(soc_pct)                    AS soc_max,
                   MIN(soc_pct)                    AS soc_min,
                   COUNT(*)                        AS n_samples
            FROM telemetry_1min_clean
            WHERE system_id = '{system_id}'
            GROUP BY date ORDER BY date
        """).df()
    df["date"] = pd.to_datetime(df["date"])
    df.loc[df["n_samples"] < 60, "spread"] = float("nan")
    return df


@st.cache_data(ttl=3600)
def get_daily_voltage_spread(system_id: str) -> pd.DataFrame:
    """Daily pack-voltage range (max − min V per day). Higher swing
    typically tracks deeper cycling but also reflects C-rate × IR.
    """
    with connect() as con:
        df = con.sql(f"""
            SELECT date_trunc('day', timestamp)::DATE AS date,
                   MAX(voltage_v) - MIN(voltage_v)  AS spread,
                   MAX(voltage_v)                   AS v_max,
                   MIN(voltage_v)                   AS v_min,
                   COUNT(*)                         AS n_samples
            FROM telemetry_1min_clean
            WHERE system_id = '{system_id}'
              AND voltage_v IS NOT NULL
            GROUP BY date ORDER BY date
        """).df()
    df["date"] = pd.to_datetime(df["date"])
    df.loc[df["n_samples"] < 60, "spread"] = float("nan")
    return df
