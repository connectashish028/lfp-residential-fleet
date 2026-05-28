"""HTML KPI / status renderers for the Operator-Light theme.

Streamlit-native widgets (st.metric, st.dataframe) don't honour the
typography rules without heavy CSS overrides — these helpers emit
custom HTML that drops cleanly into st.markdown blocks.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import streamlit as st


def hero_bar(brand: str, badge: str) -> None:
    """Top-of-page brand strip — uppercase title left, status badge right."""
    st.markdown(
        f"""
        <div class="hero-bar">
            <div class="hero-brand">{brand}</div>
            <div class="hero-badge">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_grid(
    items: Iterable[Mapping[str, Any]],
    columns: int | None = None,
) -> None:
    """N-tile stat grid. Defaults to a 4-column layout; pass
    ``columns=3`` (or similar) when the row carries fewer tiles —
    used for the Min/Average/Max stats above the distribution charts.

    Each item dict supports the following keys:

    * ``label``     — uppercase tile label (required)
    * ``value``     — main numeric / text content (required)
    * ``unit``      — small text after the value (optional)
    * ``attention`` — bool, turns the tile amber-tinted
    * ``derived``   — bool, adds a "(derived)" badge after the label
    * ``tip``       — string, attaches a hover-ⓘ tooltip with the
                       given content. Use for the KPI definition or
                       formula. Tooltip text supports inline HTML.
    """
    cells: list[str] = []
    for it in items:
        cls = "stat-cell attention" if it.get("attention") else "stat-cell"
        unit = it.get("unit", "")
        label_extra = ""
        if it.get("derived"):
            label_extra += '<span class="stat-derived">derived</span>'
        if it.get("tip"):
            tip = it["tip"]
            label_extra += (
                f'<span class="info-tip">ⓘ'
                f'<span class="info-tip-content">{tip}</span>'
                f'</span>'
            )
        cells.append(
            f'<div class="{cls}">'
            f'  <div class="stat-label">{it["label"]}{label_extra}</div>'
            f'  <div class="stat-value">{it["value"]}'
            f'    <span class="stat-unit">{unit}</span>'
            f'  </div>'
            f'</div>'
        )
    inline_style = (
        f' style="grid-template-columns: repeat({columns}, 1fr);"'
        if columns is not None and columns != 4
        else ""
    )
    st.markdown(
        f'<div class="stat-grid"{inline_style}>{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def heading_with_tip(text: str, tip: str, level: int = 2) -> None:
    """Render an H2 (or H3) section heading with a hover-ⓘ tooltip.

    Replaces the long prose paragraph that used to sit under every
    chart — operator sees the bare heading by default, hovers the ⓘ
    when they want the definition. Less clutter, same information.
    """
    tag = f"h{level}"
    st.markdown(
        f"<{tag}>{text} "
        f'<span class="info-tip">ⓘ'
        f'<span class="info-tip-content">{tip}</span></span>'
        f"</{tag}>",
        unsafe_allow_html=True,
    )


def heading_with_kpi_and_tip(
    text: str,
    kpi_text: str,
    tip: str,
    level: int = 2,
) -> None:
    """Section heading with an inline KPI badge plus hover-ⓘ tooltip.

    Lets us drop the redundant 4-tile KPI strip when the same numbers
    can sit beside their chart's title — easier to scan, no duplication.
    """
    tag = f"h{level}"
    st.markdown(
        f"<{tag}>{text} "
        f'<span class="kpi-inline">{kpi_text}</span> '
        f'<span class="info-tip">ⓘ'
        f'<span class="info-tip-content">{tip}</span></span>'
        f"</{tag}>",
        unsafe_allow_html=True,
    )


def pill(level: str, text: str | None = None) -> str:
    """Return an HTML status pill — for embedding inside tables / rows.

    Level is one of ``healthy`` / ``watch`` / ``critical``.
    """
    cls = {
        "healthy":  "pill pill-healthy",
        "watch":    "pill pill-watch",
        "critical": "pill pill-critical",
    }.get(level, "pill")
    return f'<span class="{cls}">{text or level}</span>'


def finding_callout(label: str, title: str, body_html: str) -> None:
    """Coloured panel used for "Recent finding" hero on the Overview."""
    st.markdown(
        f"""
        <div class="finding-callout">
            <div class="label">{label}</div>
            <h3>{title}</h3>
            <div>{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fleet_status_grid(rows: list[dict[str, Any]]) -> None:
    """Operator-style fleet status grid — one row per system, one
    coloured dot + short status text per metric column.

    Each ``rows`` entry must carry:

    * ``system_id``     — short identifier
    * ``location``      — geography string (e.g. "Aachen, Germany")
    * ``capacity``      — pre-formatted capacity string (e.g. "8.09 kWh")
    * For each status column, two keys ``<col>_color`` (one of
      ``green`` / ``yellow`` / ``red`` / ``grey``) and ``<col>_text``
      (display string). Required columns: ``perf``, ``safety``,
      ``status``.

    The visual idiom is intentionally borrowed from operator-grade
    fleet dashboards: the eye scans down a column for the colour
    pattern, the worst dot in any row tells you which rack needs
    attention.
    """
    head = (
        "<thead><tr>"
        "<th>System</th>"
        "<th>Performance Status</th>"
        "<th>Safety Index</th>"
        "<th>Status</th>"
        "</tr></thead>"
    )
    color_class = {
        "green":  "fleet-dot-green",
        "yellow": "fleet-dot-yellow",
        "red":    "fleet-dot-red",
        "grey":   "fleet-dot-grey",
    }

    def _cell(color: str, text: str) -> str:
        cls = color_class.get(color, "fleet-dot-grey")
        return (
            "<td>"
            f'<span class="fleet-dot {cls}"></span>'
            f'<div class="fleet-cell-label">{text}</div>'
            "</td>"
        )

    body_rows: list[str] = []
    for r in rows:
        body_rows.append(
            "<tr>"
            "<td>"
            f'<span class="system-name">{r["system_id"]}</span>'
            f'<span class="system-meta">{r.get("location", "—")}</span>'
            f'<span class="system-meta">{r["capacity"]}</span>'
            "</td>"
            + _cell(r["perf_color"],   r["perf_text"])
            + _cell(r["safety_color"], r["safety_text"])
            + _cell(r["status_color"], r["status_text"])
            + "</tr>"
        )
    st.markdown(
        f'<table class="fleet-status">{head}<tbody>{"".join(body_rows)}</tbody></table>',
        unsafe_allow_html=True,
    )
