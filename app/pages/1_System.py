"""System deep-dive — per-rack telemetry, KPIs, alerts.

Pick a rack from the sidebar selector. Layout:

1. Identity strip — capacity / voltage / cells / install date / mfr
2. KPI strip — 30-day median RTE, EFC YTD, coverage %, idle %
3. Daily RTE chart (full tenure)
4. Telemetry tabs — SoC · Thermal · Dispatch
5. Alerts table — sortable, with operator recommendations
6. (ID17 only) Notable findings card
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from _components import alerts, charts, data, kpis, theme

from bess_fleet import recommendations

st.set_page_config(page_title="System · BESS Fleet Health",
                   layout="wide", initial_sidebar_state="expanded")
theme.inject(st)


# ── Page header + system picker ───────────────────────────────────────
identity = data.get_identity()
active_status = data.get_active_status().set_index("system_id")

kpis.hero_bar(
    brand="System · deep dive",
    badge="alerts · telemetry · KPIs",
)

st.markdown("<h1>System deep dive</h1>", unsafe_allow_html=True)

# System picker (left) + global time-window picker (right). The
# window value drives every chart on the page so the operator sets
# context once.
WINDOW_OPTIONS = {
    "Last 30 days":  30,
    "Last 45 days":  45,
    "Last 90 days":  90,
    "Last 1 year":   365,
}

sys_col, _spacer, time_col = st.columns([1.4, 1.6, 1.0])
with sys_col:
    default_sid = "ID16" if "ID16" in data.SYSTEMS else data.SYSTEMS[0]
    sid = st.selectbox(
        "System",
        options=data.SYSTEMS,
        index=data.SYSTEMS.index(default_sid),
        format_func=lambda s: (
            f"{s} · {identity.loc[identity['system_id']==s, 'capacity_kwh'].iloc[0]:.2f} kWh"
            + (" · retired" if not active_status.loc[s, 'is_active'] else "")
        ),
    )
with time_col:
    window_label = st.selectbox(
        "Time window",
        options=list(WINDOW_OPTIONS.keys()),
        index=0,   # default to Last 30 days
    )
window_days: int = WINDOW_OPTIONS[window_label]
window_short = window_label.replace("Last ", "")  # for KPI badges

ident_row = identity[identity["system_id"] == sid].iloc[0]
sys_active = active_status.loc[sid]

# Retirement notice — surface that "30-day window" is relative to this
# rack's own last sample so the user isn't surprised by old timestamps
if not bool(sys_active["is_active"]):
    last_seen = pd.Timestamp(sys_active["last_seen"]).strftime("%Y-%m-%d")
    st.warning(
        f"⚐ {sid} retired — last telemetry sample {last_seen} "
        f"({int(sys_active['days_since_seen'])} days ago vs the fleet's "
        f"most-recent sample). All KPIs below show this rack's final "
        f"30 days, not the fleet's last 30 days.",
        icon="ℹ",
    )


# ── Identity strip ────────────────────────────────────────────────────
kpis.stat_grid([
    {"label": "Capacity",      "value": f"{ident_row['capacity_kwh']:.2f}", "unit": "kWh"},
    {"label": "Nominal voltage", "value": f"{ident_row['voltage_nominal_v']:.0f}", "unit": "V"},
    {"label": "Cells in series", "value": int(ident_row['cells_series']), "unit": ""},
    {"label": "Install date",    "value": pd.to_datetime(ident_row['install_date']).strftime("%Y-%m"), "unit": ""},
])


# ── KPI strip — 30-day windows relative to this rack's own last sample
kpi_df = data.get_daily_kpis()
sys_kpis = kpi_df[kpi_df["system_id"] == sid].copy()
if sys_kpis.empty:
    st.warning(f"No daily KPI rows for {sid}. Run build_daily_kpis.py.")
    st.stop()

# Recent-window stats — driven by the global time-window picker
_cutoff_recent = sys_kpis["date"].max() - pd.Timedelta(days=window_days)
recent_window = sys_kpis[sys_kpis["date"] > _cutoff_recent]

rte_recent = recent_window["rte"].median()
daily_efc_recent = recent_window["efc"].median()

# ── Usable & Recoverable Energy — window-scoped breakdown ────────────
kpis.heading_with_tip(
    "Usable & Recoverable Energy",
    tip=(
        "Both views below show <b>only days that passed the four-"
        "condition RTE gate</b> — same population as the Daily-RTE "
        "chart further down. Gate-failed days are hidden here; they "
        "still appear in the SoC / Voltage spread chart above so the "
        "operator keeps diagnostic visibility on partial cycles."
        "<br><br><b>Left chart</b> — daily energy stack:"
        "<br>· <b>Usable</b> — energy out (delivered to load)"
        "<br>· <b>Cycle loss</b> — energy in − energy out (RTE inefficiency)"
        "<br>· <b>Missing</b> — fraction of day uncovered × median daily kWh"
        "<br><br><b>Right donut</b> — aggregate of the same surviving "
        "days. Implied RTE (usable / (usable + cycle loss)) matches the "
        "Daily-RTE median exactly."
    ),
)

# Build the breakdown frame for the donut. The left-side bar chart
# still shows every day for diagnostic visibility, but the donut's
# aggregate totals are computed over **only the days that passed the
# four-condition RTE gate** (rte IS NOT NULL). That keeps the
# donut's implied RTE (usable / (usable + cycle loss)) consistent
# with the Daily-RTE median surfaced below. Without this filter,
# partial-cycle days inflate the cycle-loss slice and the two
# numbers drift apart.
nameplate_kwh = float(ident_row["capacity_kwh"])
energy_window_all = sys_kpis.sort_values("date").tail(window_days).copy()
energy_window     = energy_window_all[energy_window_all["rte"].notna()].copy()

n_surviving = len(energy_window)
n_total     = len(energy_window_all)

energy_window["loss"] = (
    energy_window["energy_in_kwh"] - energy_window["energy_out_kwh"]
).clip(lower=0)
_typical = float(energy_window["energy_in_kwh"].median() or 0.0)
energy_window["missing"] = (
    1.0 - energy_window["coverage_pct"] / 100.0
).clip(lower=0) * _typical

total_usable  = float(energy_window["energy_out_kwh"].sum())
total_loss    = float(energy_window["loss"].sum())
total_missing = float(energy_window["missing"].sum())
total_dc      = total_usable + total_loss + total_missing

donut_period_label = f"{n_surviving} of {n_total} days"

chart_col, donut_col = st.columns([1.7, 1.3])
with chart_col:
    st.plotly_chart(
        charts.daily_energy_breakdown(sys_kpis, height=320, days=window_days),
        width="stretch", config=charts.PLOTLY_CONFIG,
    )
with donut_col:
    st.plotly_chart(
        charts.energy_breakdown_pie(
            usable_kwh=total_usable,
            loss_kwh=total_loss,
            missing_kwh=total_missing,
            height=320,
            period_label=donut_period_label,
        ),
        width="stretch", config=charts.PLOTLY_CONFIG,
    )


# ── Availability ──────────────────────────────────────────────────────
_avail_df = data.get_daily_availability()
_avail_sys = _avail_df[_avail_df["system_id"] == sid]
_avail_recent = data.compute_availability(window_days=window_days)
_avail_recent_row = _avail_recent[_avail_recent["system_id"] == sid]
_availability = (
    float(_avail_recent_row["availability_pct"].iloc[0])
    if not _avail_recent_row.empty else float("nan")
)
_avail_label = (
    f"{_availability:.1f}% · {window_short}"
    if pd.notna(_availability) else f"— · {window_short}"
)

kpis.heading_with_kpi_and_tip(
    "Availability",
    kpi_text=_avail_label,
    tip=(
        "<b>Data availability</b> per day, capped at 100 % (DST-safe) "
        "and interpolation-discounted: rows whose underlying 1-sec "
        "samples were reconstructed by Figgener's gap-filling are "
        "weighted down. <b>Green</b> bars ≥ 90 % · <b>red</b> bars "
        "&lt; 90 % (sustained outage). Window matches the global "
        "time-window picker at the top."
    ),
)
st.plotly_chart(
    charts.availability_chart(_avail_sys, height=240, days=window_days),
    width="stretch", config=charts.PLOTLY_CONFIG,
)


# ── Daily RTE — severity-coloured bars ────────────────────────────────
_rte_label = (
    f"{rte_recent * 100:.1f}% · {window_short}"
    if pd.notna(rte_recent) else f"— · {window_short}"
)
kpis.heading_with_kpi_and_tip(
    "Daily RTE",
    kpi_text=_rte_label,
    tip=(
        "Daily round-trip efficiency. Severity-coloured bars: "
        "🟢 ≥85 % healthy · 🟡 75–85 % moderate · 🔴 &lt;75 % concerning. "
        "Days that fail the confidence gate are filtered to NULL — "
        "see Methodology. The inline KPI is the median over the "
        "selected window."
    ),
)
st.plotly_chart(
    charts.daily_rte_chart(sys_kpis, height=280, days=window_days),
    width="stretch", config=charts.PLOTLY_CONFIG,
)


# ── Daily cycling — plain blue bars ───────────────────────────────────
_cyc_label = (
    f"{daily_efc_recent:.2f} EFC/day · {window_short}"
    if pd.notna(daily_efc_recent) else f"— EFC/day · {window_short}"
)
kpis.heading_with_kpi_and_tip(
    "Daily cycling · EFC/day",
    kpi_text=_cyc_label,
    tip=(
        "Daily <b>equivalent full cycles</b> = "
        "<code>throughput_kwh / (2 × capacity_kwh)</code>. Residential "
        "systems typically run 0.3–0.8 EFC/day; seasonal pattern "
        "reflects PV availability. The inline KPI is the median over "
        "the selected window."
    ),
)
st.plotly_chart(
    charts.daily_cycling_chart(sys_kpis, height=240, days=window_days),
    width="stretch", config=charts.PLOTLY_CONFIG,
)


# ── Telemetry tabs ────────────────────────────────────────────────────
kpis.heading_with_kpi_and_tip(
    "Telemetry",
    kpi_text=window_short,
    tip=(
        "1-minute cadence telemetry for the time window chosen at "
        "the top of the page. Tabs:"
        "<br>· <b>SoC</b> — OCV-corrected coulomb-counted state of charge"
        "<br>· <b>Thermal</b> — battery surface, room ambient, |ΔT| residual"
        "<br>· <b>Dispatch</b> — power and C-rate"
    ),
)
bounds = data.get_telemetry_bounds()
b = bounds[bounds["system_id"] == sid].iloc[0]
data_max = b["max_ts"]

# Telemetry window is driven by the global picker at the top of the
# page — one selector for the whole System view.
custom_end_dt   = data_max
custom_start_dt = data_max - pd.Timedelta(days=window_days)

tele = data.get_telemetry(sid, custom_start_dt, custom_end_dt)

# Events visible in the current telemetry window — split by channel so
# each chart gets only its relevant alerts.
_all_events = data.get_threshold_events()
window_events = _all_events[
    (_all_events["system_id"] == sid)
    & (_all_events["start"] >= custom_start_dt)
    & (_all_events["start"] <  custom_end_dt)
].copy()

soc_event_rules = {"cell_v_low", "cell_v_deep_undervolt",
                   "cell_v_overcharge", "system_dark"}
thermal_event_channels = {"temperature_c", "thermal_delta_c"}
dispatch_event_channels = {"c_rate"}

events_soc      = window_events[window_events["rule_id"].isin(soc_event_rules)]
events_thermal  = window_events[window_events["channel"].isin(thermal_event_channels)]
events_dispatch = window_events[window_events["channel"].isin(dispatch_event_channels)]


# Alert-detail UI helpers (`render_alert_detail`,
# `maybe_show_clicked_event`, `alert_picker_and_detail`) live in
# `_components/alerts.py` so this page reads as composition rather
# than implementation.


if tele.empty:
    st.info("No telemetry in the selected window.")
else:
    tab_soc, tab_thermal, tab_dispatch = st.tabs(
        ["SoC / V Spread", "Thermal", "Dispatch"],
    )
    with tab_soc:
        # Daily spread view — operator-readable interpretation of the
        # container-imbalance idea (single-pack adaptation) for single-pack systems. The
        # toggle picks the underlying signal; bars are coloured by
        # cycling-depth severity for SoC, plain for voltage.
        spread_choice = st.radio(
            "View by",
            options=["SoC Spread", "Voltage Spread"],
            horizontal=True,
            key=f"spread_view_{sid}",
        )
        win_start = pd.Timestamp(custom_start_dt).date()
        win_end   = pd.Timestamp(custom_end_dt).date()
        if spread_choice == "SoC Spread":
            soc_spread = data.get_daily_soc_spread(sid)
            soc_spread = soc_spread[
                (soc_spread["date"].dt.date >= win_start)
                & (soc_spread["date"].dt.date <  win_end)
            ]
            st.markdown(
                '<p style="color:rgba(0,0,0,0.55); font-family:\'JetBrains Mono\','
                'monospace; font-size:0.75rem; margin-top:-0.4rem;">'
                "Daily SoC range (max − min). "
                "🟢 &lt;30 % light cycling · 🟡 30–60 % moderate · 🔴 ≥60 % deep cycling."
                "</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                charts.daily_soc_spread(soc_spread, height=320),
                width="stretch", config=charts.PLOTLY_CONFIG,
            )
        else:
            volt_spread = data.get_daily_voltage_spread(sid)
            volt_spread = volt_spread[
                (volt_spread["date"].dt.date >= win_start)
                & (volt_spread["date"].dt.date <  win_end)
            ]
            st.markdown(
                '<p style="color:rgba(0,0,0,0.55); font-family:\'JetBrains Mono\','
                'monospace; font-size:0.75rem; margin-top:-0.4rem;">'
                "Daily pack-voltage range (max − min). Tracks deep cycling "
                "and current × IR — a sustained rise here without a "
                "cycling-depth change can signal rising internal resistance."
                "</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                charts.daily_voltage_spread(volt_spread, height=320),
                width="stretch", config=charts.PLOTLY_CONFIG,
            )
        # SoC distribution by state — where does the rack sit when
        # idle vs when working? Min/Avg/Max stat row above the chart
        # mirrors the the "Overview of SoC Distribution" view.
        st.markdown(
            "<p style=\"font-family:'JetBrains Mono',monospace; "
            "font-size:0.78rem; color:rgba(0,0,0,0.55); "
            "letter-spacing:0.08em; text-transform:uppercase; "
            "margin-top:2.5rem; margin-bottom:0.8rem;\">"
            "SoC distribution · resting vs operating"
            "</p>",
            unsafe_allow_html=True,
        )
        _soc = tele["soc_pct"].dropna()
        if not _soc.empty:
            kpis.stat_grid([
                {"label": "Min SoC",     "value": f"{_soc.min():.1f}",  "unit": "%"},
                {"label": "Average SoC", "value": f"{_soc.mean():.1f}", "unit": "%"},
                {"label": "Max SoC",     "value": f"{_soc.max():.1f}",  "unit": "%"},
            ], columns=3)
        st.plotly_chart(
            charts.state_distribution(
                tele, "soc_pct", label="SoC", unit="%",
                height=320, x_range=(0, 100),
            ),
            width="stretch", config=charts.PLOTLY_CONFIG,
        )

        # Surface voltage events (still useful here for the SoC tab)
        ev_state = None
        alerts.alert_picker_and_detail(events_soc, f"soc_pick_{sid}", ev_state)
    with tab_thermal:
        # Window-scoped stats — |ΔT| magnitude, since signed ΔT can
        # mislead for systems where the ambient sensor sits warmer
        # than the cell (ID17's fan-exhaust placement).
        _abs_dt = tele["thermal_delta_c"].abs()
        _mean_dt = float(_abs_dt.mean())  if _abs_dt.notna().any() else float("nan")
        _max_dt  = float(_abs_dt.max())   if _abs_dt.notna().any() else float("nan")
        _mean_tbat = float(tele["temperature_c"].mean()) if tele["temperature_c"].notna().any() else float("nan")
        st.markdown(
            "<p style=\"font-family:'JetBrains Mono',monospace; "
            "font-size:0.78rem; color:rgba(0,0,0,0.70); margin-top:-0.4rem;\">"
            f"<span class='kpi-inline'>Mean |ΔT| {_mean_dt:.2f} °C</span> "
            f"<span class='kpi-inline'>Max |ΔT| {_max_dt:.2f} °C</span> "
            f"<span class='kpi-inline'>Mean T_bat {_mean_tbat:.1f} °C</span>"
            "</p>",
            unsafe_allow_html=True,
        )
        ev_state = st.plotly_chart(
            charts.thermal_panel(tele, events=events_thermal, height=340),
            width="stretch", config=charts.PLOTLY_CONFIG_SELECTABLE,
            on_select="rerun", selection_mode=["points"],
            key=f"thermal_chart_{sid}",
        )

        # Temperature distribution by state — operating typically
        # shifts a few degrees above resting from I²R losses
        st.markdown(
            "<p style=\"font-family:'JetBrains Mono',monospace; "
            "font-size:0.78rem; color:rgba(0,0,0,0.55); "
            "letter-spacing:0.08em; text-transform:uppercase; "
            "margin-top:2.5rem; margin-bottom:0.8rem;\">"
            "T_bat distribution · resting vs operating"
            "</p>",
            unsafe_allow_html=True,
        )
        _tbat = tele["temperature_c"].dropna()
        if not _tbat.empty:
            kpis.stat_grid([
                {"label": "Min T_bat",     "value": f"{_tbat.min():.1f}",  "unit": "°C"},
                {"label": "Average T_bat", "value": f"{_tbat.mean():.1f}", "unit": "°C"},
                {"label": "Max T_bat",     "value": f"{_tbat.max():.1f}",  "unit": "°C"},
            ], columns=3)
        st.plotly_chart(
            charts.state_distribution(
                tele, "temperature_c", label="T_bat", unit="°C", height=340,
            ),
            width="stretch", config=charts.PLOTLY_CONFIG,
        )

        alerts.alert_picker_and_detail(events_thermal, f"thermal_pick_{sid}", ev_state)
    with tab_dispatch:
        _mean_c = float(tele["c_rate"].mean()) if tele["c_rate"].notna().any() else float("nan")
        _peak_c = float(tele["c_rate"].max())  if tele["c_rate"].notna().any() else float("nan")
        _mean_p = float(tele["power_kw"].abs().mean()) if tele["power_kw"].notna().any() else float("nan")
        st.markdown(
            "<p style=\"font-family:'JetBrains Mono',monospace; "
            "font-size:0.78rem; color:rgba(0,0,0,0.70); margin-top:-0.4rem;\">"
            f"<span class='kpi-inline'>Mean C-rate {_mean_c:.3f} C</span> "
            f"<span class='kpi-inline'>Peak C-rate {_peak_c:.2f} C</span> "
            f"<span class='kpi-inline'>Mean |Power| {_mean_p:.2f} kW</span>"
            "</p>",
            unsafe_allow_html=True,
        )
        ev_state = st.plotly_chart(
            charts.dispatch_panel(tele, events=events_dispatch, height=320),
            width="stretch", config=charts.PLOTLY_CONFIG_SELECTABLE,
            on_select="rerun", selection_mode=["points"],
            key=f"dispatch_chart_{sid}",
        )

        # C-rate distribution by state — resting clusters near zero
        # (almost trivially), operating shows the duty-intensity
        # distribution. The contrast itself is the signal.
        st.markdown(
            "<p style=\"font-family:'JetBrains Mono',monospace; "
            "font-size:0.78rem; color:rgba(0,0,0,0.55); "
            "letter-spacing:0.08em; text-transform:uppercase; "
            "margin-top:2.5rem; margin-bottom:0.8rem;\">"
            "C-rate distribution · resting vs operating"
            "</p>",
            unsafe_allow_html=True,
        )
        _crate = tele["c_rate"].dropna()
        if not _crate.empty:
            kpis.stat_grid([
                {"label": "Min C-rate",     "value": f"{_crate.min():.3f}",  "unit": "C"},
                {"label": "Average C-rate", "value": f"{_crate.mean():.3f}", "unit": "C"},
                {"label": "Max C-rate",     "value": f"{_crate.max():.3f}",  "unit": "C"},
            ], columns=3)
        st.plotly_chart(
            charts.state_distribution(
                tele, "c_rate", label="C-rate", unit="C", height=340,
            ),
            width="stretch", config=charts.PLOTLY_CONFIG,
        )

        alerts.alert_picker_and_detail(events_dispatch, f"dispatch_pick_{sid}", ev_state)


# ── Alerts table with operator recommendations ────────────────────────
kpis.heading_with_tip(
    "Alerts · with recommendations",
    tip=(
        "Threshold-event log for this rack, newest first. Each row "
        "is a sustained rule violation; the <b>Recommended action</b> "
        "column maps the rule_id and peak value to an operator-readable "
        "next step from "
        "<code>src/bess_fleet/recommendations.py</code>."
    ),
)

events = data.get_threshold_events()
sys_events = events[events["system_id"] == sid].copy()
_ncells_sys = int(ident_row["cells_series"])

if sys_events.empty:
    st.info(f"No threshold events for {sid} on record.")
else:
    # Top 5 most recent. Each row is a clickable card; the button
    # opens a modal with the day's time-series + threshold band so
    # the alert's *cause* is visible, not just its existence.
    top5 = sys_events.sort_values("start", ascending=False).head(5).reset_index(drop=True)

    @st.dialog("Alert details", width="large")
    def _show_alert(row_idx: int) -> None:
        row = top5.iloc[row_idx]
        rec = recommendations.for_threshold_event(
            row["rule_id"], float(row["peak_value"]), float(row["duration_min"]),
        )
        pill_level = "critical" if rec["severity"] == "critical" else "watch"

        st.markdown(
            f"### {row['rule_id']} · "
            f"{pd.Timestamp(row['start']).strftime('%Y-%m-%d %H:%M')}"
        )
        st.markdown(
            f"{kpis.pill(pill_level, rec['severity'])} "
            "&nbsp;&nbsp;"
            "<span style=\"font-family:'JetBrains Mono',monospace; "
            "font-size:0.82rem; color:rgba(0,0,0,0.65);\">"
            f"Channel <code>{row['channel']}</code> · "
            f"Peak {float(row['peak_value']):.2f} · "
            f"Duration {float(row['duration_min']):.0f} min"
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Recommended action.** {rec['action']}")

        # Day-of-event telemetry slice
        day_start = pd.Timestamp(row["start"]).normalize()
        day_end   = day_start + pd.Timedelta(days=1)
        day_tele  = data.get_telemetry(sid, day_start, day_end)
        fig = charts.alert_day_chart(
            day_tele,
            rule_id=row["rule_id"],
            event_start=pd.Timestamp(row["start"]),
            event_end=pd.Timestamp(row["end"]),
            cells_series=_ncells_sys,
            height=340,
        )
        if fig is not None:
            st.plotly_chart(fig, width="stretch", config=charts.PLOTLY_CONFIG)
        else:
            st.info("No chart mapping for this rule_id — see alerts table.")

    st.markdown(
        f'<p style="color:rgba(0,0,0,0.55); font-family:\'JetBrains Mono\',monospace;'
        f'font-size:0.78rem;">'
        f"showing 5 most recent · {len(sys_events):,} events on record"
        "</p>",
        unsafe_allow_html=True,
    )

    for _idx in range(len(top5)):
        _row = top5.iloc[_idx]
        _rec = recommendations.for_threshold_event(
            _row["rule_id"], float(_row["peak_value"]), float(_row["duration_min"]),
        )
        _pill = kpis.pill(
            "critical" if _rec["severity"] == "critical" else "watch",
            _rec["severity"],
        )
        with st.container(border=True):
            cc = st.columns([1.8, 1.4, 3.6, 0.8])
            with cc[0]:
                st.markdown(
                    f"<span style=\"font-family:'JetBrains Mono',monospace; "
                    f"font-size:0.95rem; font-weight:500;\">{_row['rule_id']}</span><br>"
                    f"<span style=\"color:rgba(0,0,0,0.55); "
                    f"font-family:'JetBrains Mono',monospace; font-size:0.76rem;\">"
                    f"{pd.Timestamp(_row['start']).strftime('%Y-%m-%d %H:%M')}</span>",
                    unsafe_allow_html=True,
                )
            with cc[1]:
                st.markdown(
                    f"{_pill}<br>"
                    f"<span style=\"color:rgba(0,0,0,0.55); "
                    f"font-family:'JetBrains Mono',monospace; font-size:0.76rem;\">"
                    f"Peak {float(_row['peak_value']):.2f} · "
                    f"{float(_row['duration_min']):.0f} min</span>",
                    unsafe_allow_html=True,
                )
            with cc[2]:
                st.markdown(
                    f"<span style=\"font-size:0.84rem; color:rgba(0,0,0,0.75); "
                    f"line-height:1.45;\">{_rec['action']}</span>",
                    unsafe_allow_html=True,
                )
            with cc[3]:
                if st.button("Open", key=f"alert_btn_{_idx}", width="stretch"):
                    _show_alert(_idx)


# ── ID17 Recent Finding callout hidden for now. Source in git history.


