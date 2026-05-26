"""Unit tests for the recommendations engine.

The engine is the operator-facing surface of the alerting pipeline:
every threshold-event row and every daily-KPI excursion gets mapped to
one of these recommendations. Voice rules and severity assignment must
be locked down, because operators read these at 2 a.m.
"""
from __future__ import annotations

import pytest

from bess_fleet.recommendations import (
    Recommendation,
    for_high_dt,
    for_rte_drop,
    for_threshold_event,
)

# ─── for_threshold_event ─────────────────────────────────────────────


class TestForThresholdEvent:
    """Every rule_id emitted by detect_threshold_events.py must map to a
    valid (severity, action) pair, with the peak and duration values
    substituted into the action template."""

    def test_thermal_critical_isolates_rack(self) -> None:
        rec = for_threshold_event("t_bat_critical", peak_value=62.4, duration_min=3)
        assert rec["severity"] == "critical"
        assert "Isolate rack" in rec["action"]
        assert "62" in rec["action"]  # peak rendered

    def test_overcharge_is_critical(self) -> None:
        """BMS failing to enforce upper cutoff is a critical safety event."""
        rec = for_threshold_event("cell_v_overcharge", peak_value=3.71, duration_min=2)
        assert rec["severity"] == "critical"
        assert "3.71" in rec["action"]
        assert "BMS" in rec["action"]

    def test_cell_v_low_vs_deep_undervolt_severity_split(self) -> None:
        """Two-tier undervolt — warning when BMS is working at cutoff,
        critical when it's failing to protect."""
        warn = for_threshold_event("cell_v_low",            peak_value=2.4, duration_min=10)
        crit = for_threshold_event("cell_v_deep_undervolt", peak_value=1.8, duration_min=10)
        assert warn["severity"] == "warning"
        assert crit["severity"] == "critical"

    def test_c_rate_impossible_flags_sensor_failure(self) -> None:
        rec = for_threshold_event("c_rate_impossible", peak_value=1.4, duration_min=1)
        assert rec["severity"] == "critical"
        assert "physically impossible" in rec["action"]

    def test_capacity_stratified_delta_t_rules_exist(self) -> None:
        """The 8 kWh and 9 kWh groups each have their own ΔT rule."""
        r8 = for_threshold_event("delta_t_high_8kwh", peak_value=11.2, duration_min=20)
        r9 = for_threshold_event("delta_t_high_9kwh", peak_value=16.3, duration_min=20)
        assert r8["severity"] == "warning"
        assert r9["severity"] == "warning"
        assert "8 kWh" in r8["action"]
        assert "9 kWh" in r9["action"]

    def test_unknown_rule_falls_back_to_info(self) -> None:
        """The dashboard must stay robust when a new rule_id appears
        before the recommendations engine has been extended."""
        rec = for_threshold_event("brand_new_rule", peak_value=1.0, duration_min=1)
        assert rec["severity"] == "info"
        assert "brand_new_rule" in rec["action"]

    def test_action_is_a_single_short_sentence(self) -> None:
        """Voice rule: recommendations are one sentence, full stop. No
        operator should ever see a multi-paragraph wall of text."""
        rec = for_threshold_event("t_bat_warm", peak_value=46.0, duration_min=8)
        # Heuristic: trailing punctuation and no embedded newlines
        assert rec["action"].endswith(".")
        assert "\n" not in rec["action"]
        # Under ~160 chars keeps it scannable on the dashboard alerts table
        assert len(rec["action"]) < 200


# ─── for_rte_drop ────────────────────────────────────────────────────


class TestForRteDrop:
    """RTE-drop tiering — magnitude_pp is the *fractional* drop
    (0.07 = 7 pp). The fixture below tests the cutoffs exactly."""

    @pytest.mark.parametrize("magnitude_pp,expected_severity", [
        (0.15, "critical"),  # well above 10pp cutoff
        (0.10, "critical"),  # exactly on cutoff (>=)
        (0.07, "warning"),   # between 5 and 10
        (0.05, "warning"),   # exactly on warning cutoff (>=)
        (0.03, "info"),      # between 2 and 5
        (0.02, "info"),      # exactly on info cutoff (>=)
    ])
    def test_severity_tier(self, magnitude_pp: float, expected_severity: str) -> None:
        rec = for_rte_drop(magnitude_pp)
        assert rec["severity"] == expected_severity

    def test_tiny_drop_is_no_op(self) -> None:
        """Below the info threshold → no recommendation, just monitor."""
        rec = for_rte_drop(0.01)
        assert rec["severity"] == "info"
        assert "no action" in rec["action"]

    def test_negative_magnitude_is_treated_as_absolute(self) -> None:
        """Some upstream code passes signed values — defensive abs()
        keeps the rule semantics intact."""
        rec_pos = for_rte_drop(0.08)
        rec_neg = for_rte_drop(-0.08)
        assert rec_pos == rec_neg


# ─── for_high_dt ─────────────────────────────────────────────────────


class TestForHighDt:
    """Thermal-residual recommendations — ΔT in °C."""

    @pytest.mark.parametrize("mean_dt_c,expected_severity", [
        (12.0, "critical"),
        (10.0, "critical"),  # exactly on cutoff
        (8.0,  "warning"),
        (7.0,  "warning"),   # exactly on cutoff
        (6.0,  "info"),
        (5.0,  "info"),      # exactly on cutoff
    ])
    def test_severity_tier(self, mean_dt_c: float, expected_severity: str) -> None:
        rec = for_high_dt(mean_dt_c)
        assert rec["severity"] == expected_severity

    def test_normal_dt_is_no_op(self) -> None:
        """ΔT below 5 °C → no recommendation."""
        rec = for_high_dt(3.2)
        assert rec["severity"] == "info"
        assert "no action" in rec["action"]


# ─── Cross-cutting contract ──────────────────────────────────────────


class TestRecommendationContract:
    """Every recommendation, regardless of rule, must conform to the
    shared Recommendation TypedDict and the voice rules."""

    def test_returns_typed_dict_with_two_keys(self) -> None:
        rec: Recommendation = for_threshold_event("t_bat_warm", 46.0, 8.0)
        assert set(rec.keys()) == {"severity", "action"}

    def test_severity_is_one_of_three_values(self) -> None:
        for rec in (
            for_threshold_event("t_bat_warm",       46.0, 8.0),
            for_rte_drop(0.15),
            for_high_dt(12.0),
        ):
            assert rec["severity"] in {"info", "warning", "critical"}
