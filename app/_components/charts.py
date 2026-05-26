"""Plotly chart builders — Operator-Light theme.

Every figure goes through :func:`_base_layout` for consistent
typography, axis styling and tooltip behaviour. The two-accent rule
(lilac for derived values, blue for measured) is honoured throughout
except on multi-system overlay charts, where colour codes *system
identity* rather than data lineage — those use :data:`theme.SYSTEM_COLOR`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as t

PLOTLY_CONFIG = {"displaylogo": False, "displayModeBar": False}

# Variant config for charts that participate in Streamlit's on_select
# flow — keep the modebar visible (auto-show on hover) so the lasso /
# box-select tools are reachable. Clicking a single point still works
# without entering a select mode, but having the tools available
# avoids confused users.
PLOTLY_CONFIG_SELECTABLE = {"displaylogo": False, "displayModeBar": "hover"}

# Severity → marker colour for event overlays on telemetry charts.
# Critical events get the red SEV_CRITICAL accent; warnings get amber.
_EVENT_COLOR: dict[str, str] = {
    "critical": t.SEV_CRITICAL,
    "warning":  t.SEV_WARNING,
    "info":     t.TEXT_50,
}


def _event_overlay(
    events: pd.DataFrame,
    y_values: list[float] | pd.Series | None = None,
    y_fixed: float | None = None,
) -> go.Scatter | None:
    """Build a clickable scatter trace of event markers.

    Each marker carries the full event row in ``customdata`` so the
    page-level ``on_select`` handler can render an alert-detail card
    when the operator clicks one.

    Parameters
    ----------
    events
        DataFrame with at least ``start``, ``rule_id``, ``severity``,
        ``peak_value``, ``duration_min``, ``channel`` columns. Must
        already be filtered to the relevant chart window + channels.
    y_values
        Per-event Y position. Use when events map cleanly onto the
        chart's Y axis (e.g. thermal events on the thermal chart use
        ``peak_value``).
    y_fixed
        Single Y position for every marker. Use when events don't map
        onto the Y axis (e.g. voltage-derived events on a SoC chart).

    Returns ``None`` when ``events`` is empty so the caller can skip
    adding the trace.
    """
    if events is None or events.empty:
        return None
    if y_values is None:
        y_values = [y_fixed] * len(events)

    colors = events["severity"].map(_EVENT_COLOR).fillna(t.SEV_WARNING).tolist()
    hover_text = events.apply(
        lambda r: (
            f"<b>{str(r['severity']).upper()}</b> · {r['rule_id']}<br>"
            f"{pd.Timestamp(r['start']).strftime('%Y-%m-%d %H:%M')}<br>"
            f"Peak: {r['peak_value']:.2f} · Duration: {r['duration_min']:.0f} min"
            "<br><i>click marker for details</i>"
        ),
        axis=1,
    ).tolist()

    return go.Scatter(
        x=events["start"], y=list(y_values), mode="markers", name="alerts",
        marker=dict(
            color=colors, size=11, symbol="circle",
            line=dict(width=1.5, color="#ffffff"),
            opacity=0.95,
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        customdata=events[[
            "rule_id", "severity", "peak_value", "duration_min",
            "start", "channel",
        ]].astype(object).values,
    )

_AXIS = dict(
    showgrid=True, gridcolor=t.BORDER, gridwidth=1,
    zeroline=False, color=t.TEXT_70,
    tickfont=dict(family="JetBrains Mono, monospace", size=11, color=t.TEXT_70),
    title_font=dict(family="JetBrains Mono, monospace", size=11, color=t.TEXT_50),
)


def _base_layout(title: str | None = None, height: int = 360) -> dict:
    layout: dict = dict(
        paper_bgcolor=t.BG, plot_bgcolor=t.BG,
        font=dict(family="Inter, sans-serif", color=t.TEXT),
        margin=dict(l=55, r=20, t=40 if title else 12, b=50),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(family="JetBrains Mono, monospace", size=10, color=t.TEXT_70),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=_AXIS, yaxis=_AXIS,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=t.HOVER_BG, bordercolor=t.BORDER,
            font=dict(family="JetBrains Mono, monospace", color="#fff", size=11),
        ),
    )
    if title:
        layout["title"] = dict(
            text=title, font=dict(size=14, color=t.TEXT_70),
            x=0, xanchor="left",
        )
    return layout


def thermal_panel(
    df: pd.DataFrame,
    events: pd.DataFrame | None = None,
    height: int = 320,
) -> go.Figure:
    """T_bat (blue), T_room (baseline dashed), |ΔT| (lilac).

    |ΔT| is plotted as a magnitude so the threshold band (and the
    operator's intuition for "how far the battery sits from room
    temperature") reads naturally. Raw signed ΔT lives in the
    database; we just display the absolute value here.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["temperature_c"], mode="lines", name="T_bat",
        line=dict(color=t.ACTUAL, width=1.3),
        hovertemplate="%{y:.1f} °C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["ambient_c"], mode="lines", name="T_room",
        line=dict(color=t.BASELINE, width=1.0, dash="dot"),
        hovertemplate="%{y:.1f} °C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["thermal_delta_c"].abs(),
        mode="lines", name="|ΔT|",
        line=dict(color=t.PREDICTION, width=1.3),
        hovertemplate="%{y:.1f} °C<extra></extra>",
    ))
    if events is not None and not events.empty:
        ev = _event_overlay(events, y_values=events["peak_value"].tolist())
        if ev is not None:
            fig.add_trace(ev)
    fig.update_layout(**_base_layout(height=height), clickmode="event+select")
    fig.update_yaxes(title_text="temperature [°C]")
    fig.update_xaxes(title_text="")
    return fig


def dispatch_panel(
    df: pd.DataFrame,
    events: pd.DataFrame | None = None,
    height: int = 320,
) -> go.Figure:
    """Power (blue) + C-rate (lilac) on twin y-axes.

    C-rate events are plotted on the right-hand (C-rate) axis at
    their ``peak_value``.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["power_kw"], mode="lines", name="Power",
        line=dict(color=t.ACTUAL, width=1.2),
        hovertemplate="%{y:.2f} kW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["c_rate"], mode="lines", name="C-rate",
        yaxis="y2", line=dict(color=t.PREDICTION, width=1.0, dash="dot"),
        hovertemplate="%{y:.3f} C<extra></extra>",
    ))
    if events is not None and not events.empty:
        ev = _event_overlay(events, y_values=events["peak_value"].tolist())
        if ev is not None:
            ev.yaxis = "y2"
            fig.add_trace(ev)
    layout = _base_layout(height=height)
    layout["yaxis"] = {**_AXIS, "title_text": "kW"}
    layout["yaxis2"] = {**_AXIS, "title_text": "C-rate", "overlaying": "y",
                        "side": "right", "showgrid": False}
    layout["clickmode"] = "event+select"
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="")
    return fig


def energy_breakdown_pie(
    usable_kwh: float,
    loss_kwh: float,
    missing_kwh: float,
    aging_kwh: float,
    height: int = 280,
    period_label: str = "60 d",
) -> go.Figure:
    """Donut chart matching the daily stacked-bar categories.

    Same four buckets, same colors. The hole carries the aggregate
    energy figure for the period; legend on the right surfaces the
    absolute kWh per bucket so the reader doesn't have to hover.
    """
    labels = ["Usable", "Cycle loss", "Missing data", "Aging (est.)"]
    values = [
        max(0.0, float(usable_kwh)),
        max(0.0, float(loss_kwh)),
        max(0.0, float(missing_kwh)),
        max(0.0, float(aging_kwh)),
    ]
    colors = [t.SEV_HEALTHY, t.ACTUAL, t.SEV_WARNING, t.TEXT_30]
    total = sum(values)

    # Legend label rendered with the absolute kWh and the percent so
    # the reader doesn't have to hover. Empty inside-slice labels —
    # they overlap on the tiny Missing-data slice and confuse the eye.
    legend_labels = [
        f"{lab} &nbsp; {v:.1f} kWh"
        f" &nbsp; ({(v / total * 100):.1f}%)" if total > 0 else lab
        for lab, v in zip(labels, values, strict=True)
    ]

    fig = go.Figure(data=[go.Pie(
        labels=legend_labels, values=values,
        marker=dict(colors=colors, line=dict(color=t.BG, width=2)),
        hole=0.55,
        textinfo="none",
        hovertemplate="<b>%{label}</b><extra></extra>",
        sort=False, direction="clockwise",
    )])
    # Legend stacked underneath the donut so the donut can use the
    # full column width — the previous right-side legend squeezed the
    # ring into a thin sliver.
    fig.update_layout(
        paper_bgcolor=t.BG, plot_bgcolor=t.BG,
        height=height,
        margin=dict(l=20, r=20, t=10, b=10),
        showlegend=True,
        legend=dict(
            orientation="v", yanchor="middle", y=0.5,
            xanchor="left",   x=1.05,
            font=dict(family="JetBrains Mono, monospace", size=10, color=t.TEXT_70),
            bgcolor="rgba(0,0,0,0)",
            itemclick=False, itemdoubleclick=False,
        ),
        annotations=[
            dict(
                text=f"<b>{total:.1f}</b>",
                x=0.5, y=0.54, showarrow=False, xanchor="center", yanchor="middle",
                font=dict(family="JetBrains Mono, monospace", size=18, color=t.TEXT),
            ),
            dict(
                text=f"kWh · {period_label}",
                x=0.5, y=0.43, showarrow=False, xanchor="center", yanchor="middle",
                font=dict(family="JetBrains Mono, monospace", size=9, color=t.TEXT_50),
            ),
        ],
    )
    return fig


def daily_energy_breakdown(
    daily_kpis_for_system: pd.DataFrame,
    height: int = 300,
    days: int = 60,
) -> go.Figure:
    """Stacked daily energy bars — usable / cycle-loss / missing.

    Operator-style Usable & Recoverable Energy chart. For a
    single-pack residential rack we can't credibly decompose ``Aging``
    on a per-day basis, so aging surfaces only in the summary panel
    alongside this chart. Daily stack is honest about what the data
    actually carries:

    * **Usable** (green)   = ``energy_out_kwh``       (delivered)
    * **Cycle loss** (blue) = ``energy_in − energy_out`` (RTE inefficiency)
    * **Missing** (amber)  = ``(1 − coverage/100) × median daily in_kwh``
    """
    df = daily_kpis_for_system.sort_values("date").tail(days).copy()
    df["loss"] = (df["energy_in_kwh"] - df["energy_out_kwh"]).clip(lower=0)
    typical_daily = float(df["energy_in_kwh"].median() or 0.0)
    df["missing"] = (1.0 - df["coverage_pct"] / 100.0).clip(lower=0) * typical_daily

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["energy_out_kwh"], name="Usable",
        marker_color=t.SEV_HEALTHY,
        hovertemplate="%{x|%Y-%m-%d}<br>Usable %{y:.2f} kWh<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df["loss"], name="Cycle loss",
        marker_color=t.ACTUAL,
        hovertemplate="%{x|%Y-%m-%d}<br>Loss %{y:.2f} kWh<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df["missing"], name="Missing data",
        marker_color=t.SEV_WARNING,
        hovertemplate="%{x|%Y-%m-%d}<br>Missing %{y:.2f} kWh<extra></extra>",
    ))
    fig.update_layout(**_base_layout(height=height), barmode="stack")
    fig.update_yaxes(title_text="energy [kWh]")
    fig.update_xaxes(title_text="")
    return fig


def daily_soc_spread(
    soc_spread_df: pd.DataFrame,
    height: int = 280,
    days: int | None = None,
) -> go.Figure:
    """Daily SoC range bars, colored by magnitude.

    Severity buckets (one-pack interpretation — the container-
    imbalance categories don't translate directly):

    * 🟢 <30 %  — light cycling
    * 🟡 30–60 % — moderate cycling
    * 🔴 ≥60 %  — deep cycling
    """
    df = soc_spread_df.sort_values("date").copy()
    if days:
        df = df.tail(days)

    def _color(s: float) -> str:
        if pd.isna(s):     return t.TEXT_30
        if s >= 60:        return t.SEV_CRITICAL
        if s >= 30:        return t.SEV_WARNING
        return t.SEV_HEALTHY

    colors = [_color(float(s)) for s in df["spread"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["spread"], marker_color=colors,
        hovertemplate="%{x|%Y-%m-%d}<br>SoC spread %{y:.1f} %<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_base_layout(height=height), showlegend=False)
    fig.update_yaxes(title_text="SoC spread [%]", range=[0, 100])
    fig.update_xaxes(title_text="")
    return fig


# Mapping from threshold-event `rule_id` → which telemetry channel to
# plot in the alert-detail dialog, where the threshold sits, and which
# side of the threshold the event lives on. Drives both the channel
# selection and the coloured warning band in :func:`alert_day_chart`.
ALERT_DEFS: dict[str, dict] = {
    "t_bat_warm":         {"channel": "temperature_c",   "threshold": 45.0, "side": "above", "severity": "warning",  "unit": "°C"},
    "t_bat_high":         {"channel": "temperature_c",   "threshold": 50.0, "side": "above", "severity": "warning",  "unit": "°C"},
    "t_bat_critical":     {"channel": "temperature_c",   "threshold": 60.0, "side": "above", "severity": "critical", "unit": "°C"},
    "t_bat_cold":         {"channel": "temperature_c",   "threshold":  0.0, "side": "below", "severity": "warning",  "unit": "°C"},
    "delta_t_high_8kwh":  {"channel": "thermal_delta_c", "threshold": 10.0, "side": "above", "severity": "warning",  "unit": "°C"},
    "delta_t_high_9kwh":  {"channel": "thermal_delta_c", "threshold": 15.0, "side": "above", "severity": "warning",  "unit": "°C"},
    "c_rate_above_inverter": {"channel": "c_rate",       "threshold":  0.6, "side": "above", "severity": "warning",  "unit": "C"},
    "c_rate_impossible":  {"channel": "c_rate",          "threshold":  1.0, "side": "above", "severity": "critical", "unit": "C"},
    "cell_v_overcharge":  {"channel": "cell_voltage_v",  "threshold":  3.65, "side": "above", "severity": "critical", "unit": "V"},
    "cell_v_low":         {"channel": "cell_voltage_v",  "threshold":  2.5,  "side": "below", "severity": "warning",  "unit": "V"},
    "cell_v_deep_undervolt": {"channel": "cell_voltage_v", "threshold": 2.0, "side": "below", "severity": "critical", "unit": "V"},
    "system_dark":        {"channel": "cell_voltage_v",  "threshold":  0.5,  "side": "below", "severity": "warning",  "unit": "V"},
}


def alert_day_chart(
    day_telemetry: pd.DataFrame,
    rule_id: str,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    cells_series: int = 1,
    height: int = 320,
) -> go.Figure | None:
    """Plot one day of the channel that triggered an alert, with the
    threshold band shaded and the event window highlighted.

    Pattern matches the the "Alert Analysis" panel pattern — the operator
    sees the signal crossing into the threshold band so the alert's
    *cause* is visible, not just its existence.
    """
    spec = ALERT_DEFS.get(rule_id)
    if spec is None or day_telemetry.empty:
        return None

    df = day_telemetry.copy()
    if spec["channel"] == "cell_voltage_v":
        df["cell_voltage_v"] = df["voltage_v"] / max(1, cells_series)

    ch = spec["channel"]
    if ch not in df.columns:
        return None

    # Thermal residual is signed in the raw data (T_bat − T_room),
    # which goes negative when ambient sits warmer than the cell —
    # e.g. ID17's fan-exhaust sensor placement. The threshold rules
    # fire on |ΔT|, so display the magnitude so the threshold band
    # reads naturally.
    if ch == "thermal_delta_c":
        df["thermal_delta_c"] = df["thermal_delta_c"].abs()

    series = df[ch].dropna()
    if series.empty:
        return None

    threshold = spec["threshold"]
    side      = spec["side"]
    severity  = spec["severity"]
    unit      = spec["unit"]
    band_color = t.SEV_CRITICAL if severity == "critical" else t.SEV_WARNING

    y_min = float(series.min())
    y_max = float(series.max())
    y_pad = max(0.05 * (y_max - y_min), 0.05)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[ch], mode="lines",
        line=dict(color=t.PREDICTION, width=1.6),
        hovertemplate=f"%{{x|%H:%M}}<br>%{{y:.2f}} {unit}<extra></extra>",
        showlegend=False,
    ))

    # Coloured threshold band — paints the region the channel was
    # forbidden to enter. Operator's eye lands on it immediately.
    if side == "above":
        fig.add_hrect(
            y0=threshold, y1=max(y_max + y_pad, threshold + y_pad),
            fillcolor=band_color, opacity=0.13, line_width=0,
            layer="below",
        )
    else:  # below
        fig.add_hrect(
            y0=min(y_min - y_pad, threshold - y_pad), y1=threshold,
            fillcolor=band_color, opacity=0.13, line_width=0,
            layer="below",
        )
    fig.add_hline(
        y=threshold, line_dash="dot", line_color=band_color, line_width=1.5,
        annotation_text=f"{severity.upper()} threshold · {threshold} {unit}",
        annotation_position="top right",
        annotation_font=dict(family="JetBrains Mono, monospace",
                             size=10, color=band_color),
    )

    # Event window vertical shade
    fig.add_vrect(
        x0=event_start, x1=event_end,
        fillcolor=t.TEXT_30, opacity=0.18, line_width=0, layer="below",
    )

    fig.update_layout(**_base_layout(height=height))
    axis_label = {
        "temperature_c": "battery temperature [°C]",
        "thermal_delta_c": "|ΔT| · |T_bat − T_room| [°C]",
        "c_rate": "C-rate",
        "cell_voltage_v": "cell voltage [V]",
    }.get(ch, ch)
    fig.update_yaxes(title_text=axis_label)
    fig.update_xaxes(title_text="")
    return fig


def _gaussian_kde(
    values: np.ndarray,
    x_grid: np.ndarray,
    min_bandwidth: float | None = None,
) -> np.ndarray | None:
    """Pure-numpy Gaussian KDE using Scott's bandwidth rule.

    ``min_bandwidth`` floors the bandwidth so heavily-concentrated
    distributions (e.g. C-rate at rest ≈ 0) don't collapse to a
    delta-spike that blows up the Y scale. Pass a fraction of the
    chart's X range as the floor (e.g. ``(x_max - x_min) * 0.025``).

    Returns ``None`` when there aren't enough points.
    """
    if len(values) < 5:
        return None
    std = float(np.std(values, ddof=1))
    bandwidth = 1.06 * std * len(values) ** (-1 / 5) if std > 0 else 0.0
    if min_bandwidth is not None:
        bandwidth = max(bandwidth, min_bandwidth)
    if bandwidth == 0.0:
        return None
    diffs = (x_grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2) / np.sqrt(2 * np.pi)
    return kernel.sum(axis=1) / (len(values) * bandwidth)


def _format_duration_minutes(n_minutes: int) -> str:
    """``n`` 1-min samples → human duration like ``"11d 20h"`` or
    ``"3h 42m"``."""
    days = n_minutes // (60 * 24)
    hours = (n_minutes % (60 * 24)) // 60
    minutes = n_minutes % 60
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def state_distribution(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    unit: str = "",
    height: int = 320,
    x_range: tuple[float, float] | None = None,
    n_points: int = 200,
) -> go.Figure:
    """Operator-style distribution chart — smooth KDE curves split by
    resting vs operating.

    The Y axis reads as **% of time**: KDE density per unit-x × 100,
    so the curve area integrated over the X range = ~100 %. Legend
    carries the *duration* each state was observed for, formatted as
    days + hours (with 1-min telemetry cadence: one sample = one
    minute).

    Resting is plotted in blue (ACTUAL accent), Operating in green
    (SEV_HEALTHY) — matches the colour convention used in operator dashboards where the
    state-of-charge view distinguishes the two by a continuous-
    spectrum colour pair.
    """
    fig = go.Figure()

    if "is_idle" not in df.columns or value_col not in df.columns:
        fig.add_annotation(
            text="Channel or state flag not available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family="JetBrains Mono, monospace", size=12, color=t.TEXT_50),
        )
        fig.update_layout(**_base_layout(height=height))
        return fig

    resting_vals   = df.loc[df["is_idle"],  value_col].dropna().to_numpy()
    operating_vals = df.loc[~df["is_idle"], value_col].dropna().to_numpy()

    # Determine the X grid
    if x_range is None:
        all_vals = np.concatenate([resting_vals, operating_vals])
        if all_vals.size == 0:
            fig.update_layout(**_base_layout(height=height))
            return fig
        x_min = float(np.nanmin(all_vals))
        x_max = float(np.nanmax(all_vals))
        pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        x_min, x_max = x_min - pad, x_max + pad
    else:
        x_min, x_max = x_range
    x_grid = np.linspace(x_min, x_max, n_points)
    x_span = x_max - x_min
    # Minimum bandwidth = 2.5 % of the X range. Stops near-constant
    # signals (resting C-rate ≈ 0) from producing a delta-spike that
    # blows up the Y scale.
    min_bw = x_span * 0.025
    # Y values are reported as "% of time within ±half-bin of value",
    # using 50 typical bins across the X range so the scale stays
    # interpretable across SoC (0-100), T_bat (~10), and C-rate (~0.5).
    bin_width = x_span / 50.0

    rest_density = _gaussian_kde(resting_vals,   x_grid, min_bandwidth=min_bw)
    op_density   = _gaussian_kde(operating_vals, x_grid, min_bandwidth=min_bw)

    if rest_density is not None:
        fig.add_trace(go.Scatter(
            x=x_grid, y=rest_density * bin_width * 100,
            mode="lines",
            name=f"Resting · {_format_duration_minutes(len(resting_vals))}",
            line=dict(color=t.ACTUAL, width=2.4),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.10)",
            hovertemplate=(
                f"{label} %{{x:.2f}} {unit}<br>"
                "%{y:.2f} %% of time<extra>Resting</extra>"
            ),
        ))
    if op_density is not None:
        fig.add_trace(go.Scatter(
            x=x_grid, y=op_density * bin_width * 100,
            mode="lines",
            name=f"Operating · {_format_duration_minutes(len(operating_vals))}",
            line=dict(color=t.SEV_HEALTHY, width=2.4),
            fill="tozeroy", fillcolor="rgba(21,128,61,0.10)",
            hovertemplate=(
                f"{label} %{{x:.2f}} {unit}<br>"
                "%{y:.2f} %% of time<extra>Operating</extra>"
            ),
        ))

    layout = _base_layout(height=height)
    # Add breathing room at the bottom so the X-axis title, tick
    # labels, and the legend strip don't collide.
    layout["margin"] = dict(l=55, r=20, t=12, b=110)
    # Move legend below the chart — matches the operator-dashboard reference style.
    # y=-0.45 sits comfortably under the X-axis title without the
    # legend swatches landing on tick labels.
    layout["legend"] = dict(
        orientation="h", yanchor="top", y=-0.45,
        xanchor="center", x=0.5,
        font=dict(family="JetBrains Mono, monospace", size=11, color=t.TEXT_70),
        bgcolor="rgba(0,0,0,0)",
        itemsizing="constant",
        tracegroupgap=24,
    )
    fig.update_layout(**layout)
    fig.update_xaxes(
        title_text=f"{label}" + (f" [{unit}]" if unit else ""),
        title_standoff=14,
    )
    fig.update_yaxes(title_text="% of time", title_standoff=10)
    return fig


def daily_voltage_spread(
    volt_spread_df: pd.DataFrame,
    height: int = 280,
    days: int | None = None,
) -> go.Figure:
    """Daily pack-voltage range bars. Solid blue — magnitude alone
    doesn't have a fault-vs-healthy threshold the way SoC spread does
    (depends on C-rate × IR), so we don't bucket by severity."""
    df = volt_spread_df.sort_values("date").copy()
    if days:
        df = df.tail(days)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["spread"], marker_color=t.ACTUAL,
        hovertemplate="%{x|%Y-%m-%d}<br>V spread %{y:.2f} V<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_base_layout(height=height), showlegend=False)
    fig.update_yaxes(title_text="voltage spread [V]")
    fig.update_xaxes(title_text="")
    return fig


def daily_rte_chart(
    daily_kpis_for_system: pd.DataFrame,
    height: int = 280,
    days: int | None = 60,
) -> go.Figure:
    """Daily RTE % as severity-coloured bars.

    Defaults to the last 60 days. Pass ``days=None`` for full tenure.

    Severity thresholds match the Fleet-Overview status grid:
      🟢 ≥ 85 %   healthy
      🟡 75 – 85 %  moderate
      🔴 < 75 %    concerning
    """
    df = daily_kpis_for_system.set_index("date").sort_index()
    if days:
        df = df.tail(days)
    rte = df["rte"].mul(100)

    def _color(v: float) -> str:
        if pd.isna(v):  return t.TEXT_30
        if v >= 85:     return t.SEV_HEALTHY
        if v >= 75:     return t.SEV_WARNING
        return t.SEV_CRITICAL
    colors = [_color(float(v)) for v in rte.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rte.index, y=rte.values, name="daily",
        marker=dict(color=colors), opacity=0.85,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_base_layout(height=height), showlegend=False)
    fig.update_yaxes(title_text="RTE [%]", range=[40, 100])
    fig.update_xaxes(title_text="")
    return fig


def daily_cycling_chart(
    daily_kpis_for_system: pd.DataFrame,
    height: int = 240,
    days: int | None = 60,
) -> go.Figure:
    """Daily equivalent full cycles (EFC/day) as plain blue bars.

    Defaults to the last 60 days; pass ``days=None`` for full tenure.

    No severity coloring — cycling intensity doesn't have a "good vs
    bad" threshold the way RTE does.
    """
    df = daily_kpis_for_system.set_index("date").sort_index()
    if days:
        df = df.tail(days)
    efc = df["efc"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=efc.index, y=efc.values, name="daily",
        marker=dict(color=t.ACTUAL), opacity=0.65,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} EFC/day<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_base_layout(height=height), showlegend=False)
    fig.update_yaxes(title_text="EFC / day", rangemode="tozero")
    fig.update_xaxes(title_text="")
    return fig


def availability_chart(
    daily_availability: pd.DataFrame,
    height: int = 240,
    outage_threshold_pct: float = 90.0,
    days: int | None = 60,
) -> go.Figure:
    """Daily availability bars — green when ≥ outage threshold, red
    when below. No rolling-median overlay; the colour encodes the
    operator signal.

    Defaults to the last 60 days for consistency with the other
    System-page bar charts. Pass ``days=None`` for full tenure.
    """
    df = daily_availability.sort_values("date").copy()
    if days:
        df = df.tail(days)

    colors = [
        t.SEV_CRITICAL if (pd.notna(v) and v < outage_threshold_pct)
        else t.SEV_HEALTHY
        for v in df["availability_pct"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["availability_pct"],
        marker=dict(color=colors), opacity=0.85,
        hovertemplate="%{x|%Y-%m-%d}<br>availability %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(
        y=outage_threshold_pct, line_dash="dot",
        line_color=t.SEV_WARNING, line_width=1,
        annotation_text=f"{outage_threshold_pct:.0f} % outage threshold",
        annotation_position="bottom right",
        annotation_font=dict(family="JetBrains Mono, monospace",
                             size=10, color=t.SEV_WARNING),
    )
    fig.update_layout(**_base_layout(height=height), showlegend=False)
    fig.update_yaxes(title_text="availability [%]", range=[0, 105])
    fig.update_xaxes(title_text="")
    return fig


