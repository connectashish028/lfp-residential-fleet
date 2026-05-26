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

import numpy as np
import pandas as pd
import streamlit as st

from bess_fleet.db import connect

from .data_access import (
    NOTABLE_FINDINGS,
    SYSTEMS,
    get_active_status,
    get_daily_availability,
    get_daily_kpis,
    get_identity,
    get_threshold_events,
)

# LFP open-circuit-voltage → SoC lookup. Same table as the pipeline
# ``derive_soc`` module — kept here too so the SoH helper doesn't
# need to import from ``bess_fleet.pipeline``.
_OCV_SOC_TABLE: np.ndarray = np.array([
    [0,   2.500], [1,   2.950], [5,   3.100], [10, 3.200], [20, 3.270],
    [30,  3.300], [50,  3.310], [70,  3.320], [80,  3.330], [90, 3.360],
    [95,  3.440], [99,  3.550], [100, 3.650],
])
_CLIFF_LO_V: float = 3.10   # below this, voltage→SoC lookup is reliable
_CLIFF_HI_V: float = 3.36   # above this, voltage→SoC lookup is reliable

# SoH pair-window: maximum allowed elapsed time between two consecutive
# OCV anchors that form an implied-capacity measurement. Bounded so
# current-sensor drift can't accumulate past a tolerable level
# (e.g. 0.1 A offset over 24 h ≈ 2.4 Ah, ~1.5 % of an 158 Ah pack).
_SOH_PAIR_MAX_HOURS: float = 24.0

# Minimum qualifying months before we publish a SoH summary for a system.
# Below this, the baseline is too noisy to anchor against and the
# headline number is misleading.
_SOH_MIN_MONTHS: int = 3


def _ocv_to_soc(cell_voltage_v: np.ndarray) -> np.ndarray:
    return np.interp(
        cell_voltage_v, _OCV_SOC_TABLE[:, 1], _OCV_SOC_TABLE[:, 0],
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


# ─── State of Health (placeholder estimate) ───────────────────────────
@st.cache_data(ttl=3600, show_spinner="estimating SoH…")
def compute_soh() -> pd.DataFrame:
    """Per-(system, month) State-of-Health, normalised to each rack's
    own commissioning baseline.

    Method (naive variant of the standard OCV+CC approach — Plett
    2015 ch. 8, with industry-standard per-system baseline anchoring):

    1. **Anchors.** At every OCV anchor timestamp (cell rested ≥30 min,
       ``is_soc_anchor`` True in the cleaned view), look up SoC from
       the LFP voltage curve.
    2. **Qualifying pairs.** For each consecutive anchor pair, keep
       only those where ALL of:

       * both endpoints fall in the reliable OCV cliffs
         (V < 3.10 or V > 3.36)
       * SoC swing ≥ 30 %
       * elapsed time ≤ 24 h — bounds current-sensor drift
       * ``sign(ΔAh) == sign(ΔSoC)`` (Kirchhoff sanity)

       For each, compute::

           implied_capacity_Ah = ΔAh / (ΔSoC_ocv / 100)

       Drop pairs whose implied capacity falls outside
       [0.5×, 1.5×] nameplate. Aggregate to monthly median; drop
       months with fewer than 3 qualifying pairs.
    3. **Baseline anchor.** Per-system::

           baseline_Ah = median(first 6 qualifying months)
           SoH_pct     = implied_capacity_Ah / baseline_Ah × 100

       Median (not max) — robust to noise spikes in any single early
       month.
    4. **Clip** the normalised SoH to [70, 100].
    5. **Min-data threshold.** Systems with fewer than 3 qualifying
       months produce no SoH series.

    Columns: ``system_id``, ``month``, ``soh_pct``, ``n_pairs``.
    """
    ident = get_identity()
    cap_lookup = dict(zip(ident["system_id"], ident["capacity_ah"], strict=True))
    cells_lookup = dict(zip(ident["system_id"], ident["cells_series"], strict=True))

    all_rows: list[pd.DataFrame] = []
    for sid in SYSTEMS:
        nameplate_ah = float(cap_lookup[sid])
        ncells = int(cells_lookup[sid])

        with connect() as con:
            df = con.sql(f"""
                SELECT timestamp, cell_v, cum_ah FROM (
                    SELECT
                        timestamp,
                        voltage_v / {ncells} AS cell_v,
                        SUM(current_a) OVER (ORDER BY timestamp) / 60.0 AS cum_ah,
                        is_soc_anchor
                    FROM telemetry_1min_clean
                    WHERE system_id = '{sid}'
                ) WHERE is_soc_anchor
                ORDER BY timestamp
            """).df()
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

        cell_v = df["cell_v"].to_numpy()
        df["soc_ocv"] = _ocv_to_soc(cell_v)
        df["in_cliff"] = (cell_v < _CLIFF_LO_V) | (cell_v > _CLIFF_HI_V)

        df["dsoc"] = df["soc_ocv"].diff()
        df["dah"]  = df["cum_ah"].diff()
        df["dt_hours"] = df["timestamp"].diff().dt.total_seconds() / 3600.0
        df["prev_in_cliff"] = df["in_cliff"].shift(1).fillna(False)
        df["both_in_cliff"] = df["in_cliff"] & df["prev_in_cliff"]

        good = df[
            (df["dsoc"].abs() >= 30)
            & df["both_in_cliff"]
            & (df["dt_hours"] > 0)
            & (df["dt_hours"] <= _SOH_PAIR_MAX_HOURS)
        ].copy()
        if good.empty:
            continue
        good = good[np.sign(good["dah"]) == np.sign(good["dsoc"])]
        if good.empty:
            continue

        good["implied_ah"] = good["dah"] / (good["dsoc"] / 100.0)
        good = good[(good["implied_ah"] >= 0.5 * nameplate_ah)
                    & (good["implied_ah"] <= 1.5 * nameplate_ah)]
        if good.empty:
            continue

        good["month"] = good["timestamp"].dt.to_period("M").dt.to_timestamp()
        monthly = (
            good.groupby("month")
            .agg(implied_ah=("implied_ah", "median"),
                 n_pairs=("implied_ah", "size"))
            .reset_index()
        )
        monthly = monthly[monthly["n_pairs"] >= 3].copy()
        if len(monthly) < _SOH_MIN_MONTHS:
            continue

        monthly = monthly.sort_values("month").reset_index(drop=True)
        n_base = min(6, len(monthly))
        baseline_ah = float(monthly["implied_ah"].head(n_base).median())
        monthly["soh_pct"] = (
            monthly["implied_ah"] / baseline_ah * 100.0
        ).clip(70, 100)
        monthly["system_id"] = sid
        all_rows.append(monthly[["system_id", "month", "soh_pct", "n_pairs"]])

    if not all_rows:
        return pd.DataFrame(columns=["system_id", "month", "soh_pct", "n_pairs"])
    return pd.concat(all_rows, ignore_index=True)


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


@st.cache_data(ttl=3600)
def compute_soh_summary() -> pd.DataFrame:
    """Per-system SoH headline number — latest month's estimate.

    Used by both the Overview KPI strip (fleet median across active
    systems) and the System page (per-rack latest value). Marks
    systems with too few qualifying pairs as low-confidence.
    """
    monthly = compute_soh()
    if monthly.empty:
        return pd.DataFrame(columns=[
            "system_id", "latest_soh_pct", "latest_month",
            "n_pairs_latest", "n_months", "confidence",
        ])
    rows: list[dict] = []
    for sid, sub in monthly.groupby("system_id"):
        sub = sub.sort_values("month")
        latest = sub.iloc[-1]
        rows.append({
            "system_id":      sid,
            "latest_soh_pct": float(latest["soh_pct"]),
            "latest_month":   pd.Timestamp(latest["month"]),
            "n_pairs_latest": int(latest["n_pairs"]),
            "n_months":       int(len(sub)),
            "confidence":     "high" if len(sub) >= 6 else "low",
        })
    out = pd.DataFrame(rows)
    missing = [s for s in SYSTEMS if s not in out["system_id"].values]
    if missing:
        out = pd.concat([
            out,
            pd.DataFrame([{
                "system_id":      s,
                "latest_soh_pct": float("nan"),
                "latest_month":   pd.NaT,
                "n_pairs_latest": 0,
                "n_months":       0,
                "confidence":     "none",
            } for s in missing]),
        ], ignore_index=True)
    return out.sort_values("system_id").reset_index(drop=True)
