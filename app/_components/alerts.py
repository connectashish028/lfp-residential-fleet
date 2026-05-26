"""Alert-detail UI helpers shared by the System page.

Three primitives, all operator-facing:

* :func:`render_alert_detail` — finding-callout card for a single
  event. Used by both the chart-click flow and the picker flow.
* :func:`maybe_show_clicked_event` — handles the Plotly chart-click
  payload from Streamlit's ``on_select='rerun'`` integration.
* :func:`alert_picker_and_detail` — selectbox-plus-card combo that
  sits under each telemetry chart.

Kept in a dedicated module so the System page reads as composition
rather than implementation. Pure UI: no SQL, no math.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from bess_fleet import recommendations

from . import kpis


def render_alert_detail(event_row: dict) -> None:
    """Operator-facing alert detail card. Shown when a marker is clicked
    or a picker entry is selected."""
    rec = recommendations.for_threshold_event(
        event_row["rule_id"],
        float(event_row["peak_value"]),
        float(event_row["duration_min"]),
    )
    severity = rec["severity"]
    pill_html = kpis.pill(
        "critical" if severity == "critical" else "watch",
        severity,
    )
    kpis.finding_callout(
        label=f"alert · {event_row['rule_id']}",
        title=f"{pd.Timestamp(event_row['start']).strftime('%Y-%m-%d %H:%M')} · {pill_html}",
        body_html=(
            f"<p><b>Channel</b>: <code>{event_row['channel']}</code>"
            f"&nbsp;·&nbsp;<b>Peak</b>: {float(event_row['peak_value']):.2f}"
            f"&nbsp;·&nbsp;<b>Duration</b>: {float(event_row['duration_min']):.0f} min</p>"
            f"<p><b>Recommended action.</b> {rec['action']}</p>"
        ),
    )


def maybe_show_clicked_event(event_state, source_df: pd.DataFrame) -> None:
    """If the chart's selection state has a clicked point, render the
    matching alert's detail card. Streamlit's ``on_select='rerun'``
    returns a ``PlotlyState`` dict-like object whose ``selection`` key
    carries the clicked-point payload."""
    if not event_state:
        return
    selection = event_state.get("selection") if hasattr(event_state, "get") else None
    if not selection:
        return
    points = selection.get("points") or []
    if not points:
        return
    p = points[0]
    custom = p.get("customdata")
    if not custom:
        # Fallback: look up the event by index in the source DataFrame
        # if customdata wasn't echoed back.
        idx = p.get("point_index")
        if idx is None or source_df.empty:
            return
        if idx >= len(source_df):
            return
        row = source_df.iloc[idx]
        render_alert_detail({
            "rule_id":      row["rule_id"],
            "severity":     row["severity"],
            "peak_value":   row["peak_value"],
            "duration_min": row["duration_min"],
            "start":        row["start"],
            "channel":      row["channel"],
        })
        return
    rule_id, severity, peak, dur, start_ts, channel = custom[:6]
    render_alert_detail({
        "rule_id":      rule_id,
        "severity":     severity,
        "peak_value":   peak,
        "duration_min": dur,
        "start":        start_ts,
        "channel":      channel,
    })


def alert_picker_and_detail(
    events_df: pd.DataFrame,
    picker_key: str,
    ev_state,
) -> None:
    """Below each chart: a picker that lists every alert in the window
    plus a detail card. Markers on the chart show hover text and accept
    Plotly clicks; clicks update the picker via Streamlit's on_select
    rerun. Either way the operator lands on the same detail card."""
    if events_df.empty:
        st.markdown(
            '<p style="color: rgba(0,0,0,0.45); font-family: \'JetBrains Mono\', monospace;'
            'font-size: 0.78rem;">'
            "No alerts in this window."
            "</p>",
            unsafe_allow_html=True,
        )
        return

    # Try to pre-select an event from a chart click. Streamlit's
    # PlotlyState carries the clicked point's customdata or index.
    preselected_idx: int | None = None
    if ev_state and hasattr(ev_state, "get"):
        sel = ev_state.get("selection") or {}
        pts = sel.get("points") or []
        if pts:
            pt = pts[0]
            cust = pt.get("customdata")
            if cust:
                clicked_start = pd.Timestamp(cust[4])
                match_idx = events_df.index[events_df["start"] == clicked_start]
                if len(match_idx):
                    preselected_idx = events_df.index.get_loc(match_idx[0])
            elif pt.get("point_index") is not None:
                preselected_idx = int(pt["point_index"])

    options = list(range(len(events_df)))

    def _fmt(i: int) -> str:
        r = events_df.iloc[i]
        return (
            f"{pd.Timestamp(r['start']).strftime('%Y-%m-%d %H:%M')} · "
            f"{r['rule_id']} · {r['severity']}"
        )

    chosen = st.selectbox(
        f"Pick an alert — {len(events_df)} in window",
        options=options, format_func=_fmt,
        index=preselected_idx if preselected_idx is not None else None,
        key=picker_key,
        placeholder="hover a marker for a summary, or pick one here for the full card",
    )
    if chosen is not None:
        row = events_df.iloc[chosen]
        render_alert_detail({
            "rule_id":      row["rule_id"],
            "severity":     row["severity"],
            "peak_value":   row["peak_value"],
            "duration_min": row["duration_min"],
            "start":        row["start"],
            "channel":      row["channel"],
        })
