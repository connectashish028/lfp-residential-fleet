"""Fleet Overview — entry page for the BESS fleet health dashboard.

Severity-first layout:

1. Historical-replay banner — frames the dataset window so the page
   doesn't pretend to be live.
2. Stat strip — Attention needed, Median RTE, **Median SoH** (replaces
   the noisy Mean ΔT tile from v0), Active systems.
3. "Recent finding" callout — the ID17 internal-resistance story.
4. System status table — six rows, retirement-aware. Retired systems
   show their last-seen date instead of "—" for every column.
5. Event timeline — last 90 days, severity-coloured.

Every numeric quantity referenced here comes from
:mod:`app._components.data` — single source of truth.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
from _components import data, kpis, theme

st.set_page_config(
    page_title="Fleet Overview · BESS Fleet Health",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject(st)


# ── Hero strip + historical-replay banner ─────────────────────────────
identity = data.get_identity()
window = data.get_data_window()
fleet_min = window["fleet_min"].strftime("%Y-%m")
fleet_max = window["fleet_max"].strftime("%Y-%m-%d")

kpis.hero_bar(
    brand="BESS Fleet Health · Figgener LFP residential dataset",
    badge=f"replay · {fleet_min} → {fleet_max[:7]}",
)

# Banner — explicit about "now"
st.markdown(
    f"""
    <div class="replay-banner">
      <div>
        <span class="lbl">Historical replay</span>
        &nbsp;&nbsp;<span class="val">{fleet_min} → {fleet_max}</span>
      </div>
      <div>
        <span class="lbl">"Now" anchor</span>
        &nbsp;&nbsp;<span class="val">{fleet_max}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    "<h1>Fleet Overview</h1>"
    f'<p style="color:rgba(0,0,0,0.55); font-family:\'JetBrains Mono\',monospace;'
    f"font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase;"
    f'margin-top:0.25rem;">'
    f"{len(data.SYSTEMS)} systems · {len(identity['manufacturer'].unique())} manufacturer · "
    f"LFP chemistry"
    "</p>",
    unsafe_allow_html=True,
)


# ── Inputs for the status grid ────────────────────────────────────────
# (KPI tiles + Recent-finding callout were removed for the minimal
# demo view — the systems status grid below carries the same signals,
# and the ID17 diagnostic chain now lives at the bottom of the
# System → ID17 page.)
status = data.compute_system_status(window_days=30)
soh = data.compute_soh_summary()


# ── Fleet status grid (severity dots + short status per metric) ──
st.markdown("## Systems")


def _color_rte(rte_pct: float) -> tuple[str, str]:
    """Map a 30-day median RTE % to (color, operator-readable text)."""
    if rte_pct is None or pd.isna(rte_pct):
        return "grey", "No recent data"
    if rte_pct >= 85:
        return "green", f"{rte_pct:.1f}% of energy returned"
    if rte_pct >= 75:
        return "yellow", f"{rte_pct:.1f}% of energy returned"
    return "red", f"{rte_pct:.1f}% of energy returned"


def _color_safety(critical: int, warning: int, notable: object) -> tuple[str, str]:
    """Map event counts + curated findings to a single safety pill."""
    if critical > 0:
        return "red", "Action Required"
    if warning > 0 or notable is not None:
        return "yellow", "Action Recommended"
    return "green", "Stable"


def _color_soh(soh_pct: float) -> tuple[str, str]:
    """Map normalised SoH % to (color, text). SoH is baseline-anchored
    so the thresholds are tight: a 5pp drop = yellow, 10pp drop = red."""
    if soh_pct is None or pd.isna(soh_pct):
        return "grey", "Insufficient data"
    if soh_pct >= 95:
        return "green", f"{soh_pct:.1f}%"
    if soh_pct >= 90:
        return "yellow", f"{soh_pct:.1f}%"
    return "red", f"{soh_pct:.1f}%"


_STATUS_LOOKUP = {
    "healthy":  ("green",  "Healthy"),
    "watch":    ("yellow", "Watch"),
    "critical": ("red",    "Critical"),
    "retired":  ("grey",   "Retired"),
}


cap_lookup = dict(zip(identity["system_id"], identity["capacity_kwh"], strict=True))
soh_lookup = {r["system_id"]: r for _, r in soh.iterrows()}

grid_rows: list[dict] = []
for _, r in status.iterrows():
    sid = r["system_id"]
    soh_row = soh_lookup.get(sid, {})
    soh_pct = soh_row.get("latest_soh_pct") if isinstance(soh_row, dict) else soh_row["latest_soh_pct"]

    perf_color, perf_text = _color_rte(r["rte_pct"])
    safety_color, safety_text = _color_safety(
        int(r["critical_events"]), int(r["warning_events"]), r.get("notable_finding"),
    )
    soh_color, soh_text = _color_soh(soh_pct)
    status_color, status_text = _STATUS_LOOKUP.get(r["status"], ("grey", "Unknown"))

    grid_rows.append({
        "system_id":    sid,
        "location":     "Aachen, Germany",  # Figgener residential dataset, RWTH Aachen
        "capacity":     f"{cap_lookup.get(sid, 0):.2f} kWh",
        "perf_color":   perf_color,   "perf_text":   perf_text,
        "safety_color": safety_color, "safety_text": safety_text,
        "soh_color":    soh_color,    "soh_text":    soh_text,
        "status_color": status_color, "status_text": status_text,
    })

kpis.fleet_status_grid(grid_rows)


# Tiny footer
st.markdown("---")
st.markdown(
    f'<p style="color:rgba(0,0,0,0.30); font-family:\'JetBrains Mono\',monospace;'
    f'font-size:0.72rem; letter-spacing:0.08em;">'
    f"data: Figgener et al. 2024 (open dataset) · "
    f"page generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
    f"see Methodology for the analytical playbook and KPI definitions"
    "</p>",
    unsafe_allow_html=True,
)
