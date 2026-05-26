"""Severity-tiered, operator-readable recommendations for fleet alerts.

This module turns raw alerts (threshold-event rows, daily-KPI
excursions) into one-sentence operator runbook lines. Pure Python —
no DuckDB, no Streamlit, no pandas. Easy to test, easy to swap.

The dashboard's Alerts tab on every System page maps each event row
through :func:`for_threshold_event` to produce the `Recommendation`
column. The Overview's "Attention needed" tile counts systems whose
recommendations carry `severity == "critical"` in the last 30 days.

Voice rules (these matter — operators may read them at 2 a.m.):

* **Short**: one sentence, full stop.
* **Severity-appropriate verb at the front**: *Verify · Inspect ·
  Limit · Isolate · Dispatch*. Severity escalates the verb.
* **End in a concrete action**: never "investigate further", always
  "isolate rack and inspect BMS within 24 hours".
* **No hedging**: the recommendation is the recommendation. If we
  don't know, we say so and stop.

Aligns with the *Action Required / Action Recommended* operator-runbook
pattern used in commercial fleet-health dashboards.
"""
from __future__ import annotations

from typing import Literal, TypedDict

Severity = Literal["info", "warning", "critical"]


class Recommendation(TypedDict):
    severity: Severity
    action: str


# ─── Threshold-event rule book ─────────────────────────────────────────
# Mirrors the rule_id strings emitted by
# ``scripts/detect_threshold_events.py``. Each entry is
# ``(severity, action_template)``; the template accepts the
# ``{peak}`` and ``{duration}`` placeholders filled by
# :func:`for_threshold_event` below.
_RULES: dict[str, tuple[Severity, str]] = {
    # Thermal
    "t_bat_warm": ("warning",
        "Verify cooling airflow — battery surface {peak:.0f} °C for "
        "{duration:.0f} min, above the 45 °C warning threshold."),
    "t_bat_high": ("warning",
        "Limit operation and schedule cooling inspection within 7 days "
        "— battery {peak:.0f} °C exceeds the 50 °C operational limit."),
    "t_bat_critical": ("critical",
        "Isolate rack immediately and dispatch field tech — battery "
        "{peak:.0f} °C, thermal-runaway risk."),
    "t_bat_cold": ("warning",
        "Suspend charging until cell rises above 5 °C — battery "
        "{peak:.0f} °C, lithium-plating risk on LFP at cold-charge."),
    "delta_t_high_8kwh": ("warning",
        "Inspect HVAC — thermal residual peaked {peak:.0f} °C "
        "for {duration:.0f} min on the 8 kWh hardware group."),
    "delta_t_high_9kwh": ("warning",
        "Inspect HVAC — thermal residual peaked {peak:.0f} °C "
        "for {duration:.0f} min on the 9 kWh hardware group."),
    # Electrical
    "c_rate_above_inverter": ("warning",
        "Verify against the power record — peak {peak:.2f} C above the "
        "inverter's 0.43 C physical ceiling; likely sensor glitch."),
    "c_rate_impossible": ("critical",
        "Treat derived KPIs as unreliable until checked — peak "
        "{peak:.2f} C is physically impossible; current-sensor failure."),
    "cell_v_overcharge": ("critical",
        "Isolate rack and inspect BMS — cell voltage hit {peak:.2f} V, "
        "above the 3.65 V upper cutoff. BMS not enforcing limit."),
    "cell_v_low": ("warning",
        "Monitor for repeat events — cell voltage {peak:.2f} V for "
        "{duration:.0f} min, BMS working at lower cutoff."),
    "cell_v_deep_undervolt": ("critical",
        "Isolate rack and inspect cells — cell voltage {peak:.2f} V "
        "for {duration:.0f} min, below the 2.0 V damage threshold."),
    # Operational
    "system_dark": ("warning",
        "Verify comms link and AC supply — system offline for "
        "{duration:.0f} min. No battery action required."),
}

_FALLBACK: tuple[Severity, str] = (
    "info",
    "Unknown rule '{rule_id}' — extend recommendations.py to handle it.",
)


def for_threshold_event(
    rule_id: str,
    peak_value: float,
    duration_min: float,
) -> Recommendation:
    """Map a `threshold_events` row to a `Recommendation`.

    Parameters
    ----------
    rule_id
        The rule identifier emitted by ``detect_threshold_events``.
    peak_value
        The event's `peak_value` (max of |channel|) in the channel's
        native units.
    duration_min
        Event duration in minutes.

    Returns
    -------
    A dict with `severity` and `action` keys. Falls back to
    ``severity="info"`` for unknown rule_ids — keeps the dashboard
    robust to new rules added without code-side support.
    """
    severity, template = _RULES.get(rule_id, _FALLBACK)
    action = template.format(
        peak=peak_value,
        duration=duration_min,
        rule_id=rule_id,
    )
    return Recommendation(severity=severity, action=action)


# ─── Daily-KPI excursion recommendations ───────────────────────────────
# These are produced separately because the trigger lives in the daily
# KPI table, not the threshold-events table. The Overview status table
# and the System page's "RTE alerts" tab both use these.

_RTE_TIERS: list[tuple[float, Severity, str]] = [
    (0.10, "critical",
     "Schedule load test and consider rack replacement — RTE down "
     "{magnitude_pp:.0f} pp, likely cell failure or weak module."),
    (0.05, "warning",
     "Inspect at next O&M visit — RTE down {magnitude_pp:.0f} pp, "
     "likely cell imbalance or contact-resistance growth."),
    (0.02, "info",
     "Monitor trend — RTE down {magnitude_pp:.0f} pp, within "
     "seasonal noise."),
]


def for_rte_drop(magnitude_pp: float) -> Recommendation:
    """RTE dropped by ``magnitude_pp`` percentage points vs system median.

    Pass the *positive* magnitude (an absolute value). Returns the
    most-severe tier whose threshold is met, or an "info" no-op.
    """
    m = abs(float(magnitude_pp))
    for threshold, severity, template in _RTE_TIERS:
        if m >= threshold:
            return Recommendation(
                severity=severity,
                action=template.format(magnitude_pp=m * 100),
            )
    return Recommendation(
        severity="info",
        action="RTE within tolerance — no action.",
    )


_DT_TIERS: list[tuple[float, Severity, str]] = [
    (10.0, "critical",
     "Limit operation immediately and dispatch field tech — mean ΔT "
     "{mean_dt_c:.1f} °C; cooling system failing."),
    (7.0,  "warning",
     "Inspect HVAC within 7 days — mean ΔT {mean_dt_c:.1f} °C; "
     "cooling under-spec."),
    (5.0,  "info",
     "Monitor — mean ΔT {mean_dt_c:.1f} °C is elevated; may correlate "
     "with ambient-sensor placement."),
]


def for_high_dt(mean_dt_c: float) -> Recommendation:
    """High daily-mean thermal residual."""
    for threshold, severity, template in _DT_TIERS:
        if mean_dt_c >= threshold:
            return Recommendation(
                severity=severity,
                action=template.format(mean_dt_c=mean_dt_c),
            )
    return Recommendation(
        severity="info",
        action="Thermal within tolerance — no action.",
    )
