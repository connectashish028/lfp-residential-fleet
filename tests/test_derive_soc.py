"""Unit tests for the OCV-corrected coulomb-counted SoC.

This is the headline algorithm — Plett ch. 8 hybrid: voltage anchor at
rest, current integration between anchors, linear drift correction.
The algorithm is audited in eight notebook stress tests; these unit
tests cover the cases that need to stay green on every commit.

Synthetic 1-min fixtures live in conftest.py. Each fixture starts with
≥30 minutes of rest so the rolling-30min idle detector trips exactly
one OCV anchor at the start of the dispatch window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_fleet.pipeline.derive_soc import (
    OCV_SOC_TABLE,
    derive_soc,
    ocv_to_soc,
)

# ─── ocv_to_soc — pure lookup behaviour ─────────────────────────────


class TestOcvToSoc:
    """The lookup is the bedrock of the anchor step. Any drift here
    moves every downstream SoC number."""

    def test_table_endpoints(self) -> None:
        """Exact endpoints of the table must map to 0 % and 100 %."""
        assert ocv_to_soc(np.array([2.500]))[0] == pytest.approx(0.0)
        assert ocv_to_soc(np.array([3.650]))[0] == pytest.approx(100.0)

    def test_saturates_below_table(self) -> None:
        """A voltage below 2.5 V must clamp to 0 % rather than
        extrapolating — that's the right behaviour for a relaxed cell
        reading outside the table (almost always a sensor edge case)."""
        assert ocv_to_soc(np.array([1.0]))[0] == pytest.approx(0.0)

    def test_saturates_above_table(self) -> None:
        """Symmetric — above 3.65 V clamps to 100 %."""
        assert ocv_to_soc(np.array([4.0]))[0] == pytest.approx(100.0)

    def test_monotonic_across_full_table(self) -> None:
        """OCV → SoC must be monotonically non-decreasing across the
        whole table. If this fails, the table itself has been corrupted."""
        voltages = OCV_SOC_TABLE[:, 1]
        socs = ocv_to_soc(voltages)
        diffs = np.diff(socs)
        assert (diffs >= -1e-9).all(), "OCV→SoC table is not monotonic"

    def test_plateau_interpolates_linearly(self) -> None:
        """In the LFP plateau (3.30 → 3.31 V is 30% → 50% SoC),
        an interpolated voltage between two table rows should produce
        an interpolated SoC between the two rows."""
        # Row at 3.30 V = 30 %, row at 3.31 V = 50 %. Midpoint:
        result = ocv_to_soc(np.array([3.305]))[0]
        assert 30.0 < result < 50.0


# ─── derive_soc — algorithm behaviour ───────────────────────────────


class TestDeriveSoc:
    """End-to-end checks on synthetic 1-min frames. Capacity matches
    the typical LFP residential pack from the dataset (158 Ah, 16s)."""

    CAPACITY_AH = 158.0
    CELLS_SERIES = 16  # 51 V pack / 16 cells = 3.1875 V per cell (in plateau)

    def test_output_has_soc_columns(self, synth_telemetry_resting: pd.DataFrame) -> None:
        out = derive_soc(synth_telemetry_resting, self.CAPACITY_AH, self.CELLS_SERIES)
        assert "soc_pct" in out.columns
        assert "is_soc_anchor" in out.columns

    def test_output_preserves_row_count(self, synth_telemetry_resting: pd.DataFrame) -> None:
        """The function must not drop or duplicate rows — it writes back
        into the caller's frame layout."""
        out = derive_soc(synth_telemetry_resting, self.CAPACITY_AH, self.CELLS_SERIES)
        assert len(out) == len(synth_telemetry_resting)

    def test_soc_pct_clipped_to_valid_range(self, synth_telemetry_charging: pd.DataFrame) -> None:
        """Step 5 of the algorithm: clip output to [0, 100]."""
        out = derive_soc(synth_telemetry_charging, self.CAPACITY_AH, self.CELLS_SERIES)
        assert out["soc_pct"].min() >= 0.0
        assert out["soc_pct"].max() <= 100.0

    def test_resting_signal_yields_at_least_one_anchor(
        self, synth_telemetry_resting: pd.DataFrame,
    ) -> None:
        """120 min of zero current must trigger ≥1 OCV anchor."""
        out = derive_soc(synth_telemetry_resting, self.CAPACITY_AH, self.CELLS_SERIES)
        assert int(out["is_soc_anchor"].sum()) >= 1

    def test_charging_drives_soc_upward(
        self, synth_telemetry_charging: pd.DataFrame,
    ) -> None:
        """After the rest anchor, positive current must increase SoC."""
        out = derive_soc(synth_telemetry_charging, self.CAPACITY_AH, self.CELLS_SERIES)
        # Compare last 10 min of dispatch vs the rest period
        rest_mean    = out["soc_pct"].iloc[:30].mean()
        last_10_mean = out["soc_pct"].iloc[-10:].mean()
        assert last_10_mean > rest_mean

    def test_discharging_drives_soc_downward(
        self, synth_telemetry_discharging: pd.DataFrame,
    ) -> None:
        """After the rest anchor, negative current must decrease SoC."""
        out = derive_soc(synth_telemetry_discharging, self.CAPACITY_AH, self.CELLS_SERIES)
        rest_mean    = out["soc_pct"].iloc[:30].mean()
        last_10_mean = out["soc_pct"].iloc[-10:].mean()
        assert last_10_mean < rest_mean

    def test_output_dtype_is_float32(self, synth_telemetry_resting: pd.DataFrame) -> None:
        """SoC is persisted as float32 for storage savings."""
        out = derive_soc(synth_telemetry_resting, self.CAPACITY_AH, self.CELLS_SERIES)
        assert out["soc_pct"].dtype == np.float32

    def test_function_is_pure(self, synth_telemetry_resting: pd.DataFrame) -> None:
        """Re-running on the same input must produce the same output."""
        out_a = derive_soc(synth_telemetry_resting, self.CAPACITY_AH, self.CELLS_SERIES)
        out_b = derive_soc(synth_telemetry_resting, self.CAPACITY_AH, self.CELLS_SERIES)
        pd.testing.assert_series_equal(out_a["soc_pct"], out_b["soc_pct"])

    def test_charging_rate_matches_coulomb_count(
        self, synth_telemetry_charging: pd.DataFrame,
    ) -> None:
        """Sanity check on the integration: at +5 A for 60 min on a
        158 Ah pack, ΔSoC should be roughly:
            5 A × (60/60) h / 158 Ah × 100 % ≈ 3.16 % SoC

        We assert within a 1 pp tolerance — drift correction noise is
        bounded for a clean synthetic input.
        """
        out = derive_soc(synth_telemetry_charging, self.CAPACITY_AH, self.CELLS_SERIES)
        # Skip the first 30 min (rest) and look at the dispatch window
        delta = out["soc_pct"].iloc[-1] - out["soc_pct"].iloc[29]
        expected = 5.0 * 1.0 / self.CAPACITY_AH * 100.0
        assert abs(delta - expected) < 1.0
