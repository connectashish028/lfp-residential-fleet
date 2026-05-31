"""Unit tests for the degradation-mode estimator.

The science is only as trustworthy as its primitives, so these tests pin
the numeric core against synthetic signals with *known* answers:

* a sigmoid q(V) has its dq/dV peak exactly at the sigmoid centre — the
  ICA peak finder must recover that voltage;
* a clean slow discharge must be extracted as one sweep;
* a capacity-only decline (stable peak) must read as LLI, while a
  peak-height collapse must read as LAM.

No real data, no DuckDB — fast enough for the regular suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_fleet.pipeline.degradation_modes import (
    attribute_modes,
    find_signature_peaks,
    ica_dva_curve,
    monthly_signatures,
    reconstruct_sweeps,
)


def _sigmoid_sweep(v_centre: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic discharge sweep whose q(V) is a sigmoid centred at
    ``v_centre`` — so |dq/dV| peaks exactly there."""
    v = np.linspace(3.0, 3.6, n)
    q = 50.0 / (1.0 + np.exp(-(v - v_centre) / 0.03))
    return v, q


class TestIcaDvaCurve:

    def test_ica_peak_at_known_voltage(self) -> None:
        """The ICA peak of a sigmoid q(V) must land on the sigmoid centre."""
        v_centre = 3.30
        v, q = _sigmoid_sweep(v_centre)
        curve = ica_dva_curve(v, q)
        assert not curve.empty
        peaks = find_signature_peaks(curve["v"].to_numpy(), curve["ica"].to_numpy())
        assert peaks, "expected at least one ICA peak"
        assert peaks[0]["v"] == pytest.approx(v_centre, abs=0.03)

    def test_too_short_returns_empty(self) -> None:
        v, q = _sigmoid_sweep(3.3, n=10)
        assert ica_dva_curve(v, q).empty

    def test_flat_voltage_returns_empty(self) -> None:
        """No voltage span → nothing to differentiate against."""
        v = np.full(200, 3.3)
        q = np.linspace(0, 50, 200)
        assert ica_dva_curve(v, q).empty


class TestReconstructSweeps:

    def _slow_discharge(self, capacity_ah: float = 100.0, minutes: int = 300) -> pd.DataFrame:
        """A clean 0.05C discharge that sweeps ~25 % SoC — one valid sweep."""
        ts = pd.date_range("2020-01-01", periods=minutes, freq="1min")
        current = np.full(minutes, -5.0)            # 5 A on 100 Ah = 0.05C
        voltage = np.linspace(54.0, 50.0, minutes)  # pack volts, declining
        return pd.DataFrame({
            "timestamp": ts,
            "voltage_v": voltage,
            "current_a": current,
            "temperature_c": np.full(minutes, 25.0),
        })

    def test_extracts_single_sweep(self) -> None:
        df = self._slow_discharge()
        sweeps = reconstruct_sweeps(df, capacity_ah=100.0, cells_series=16)
        assert len(sweeps) == 1
        s = sweeps[0]
        assert s.q_ah[-1] > 20.0                     # ~25 Ah moved
        assert (np.diff(s.q_ah) >= 0).all()          # monotone

    def test_fast_current_rejected(self) -> None:
        """A 0.5C discharge is above the OCV-proxy ceiling → no sweep."""
        df = self._slow_discharge()
        df["current_a"] = -50.0                     # 0.5C
        assert reconstruct_sweeps(df, capacity_ah=100.0, cells_series=16) == []

    def test_short_swing_rejected(self) -> None:
        """Too little SoC swing to resolve peaks → no sweep."""
        df = self._slow_discharge(minutes=120)      # ~10 Ah = 0.10C-span only
        assert reconstruct_sweeps(df, capacity_ah=100.0, cells_series=16) == []


class TestAttributeModes:

    def _monthly(self, **overrides: np.ndarray) -> pd.DataFrame:
        n = 9
        base = {
            "month": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "n_sweeps": np.full(n, 10),
            "anchored_cap_ah": np.full(n, 50.0),
            "cap_cov": np.full(n, 0.05),
            "main_peak_v": np.full(n, 3.30),
            "main_peak_height": np.full(n, 100.0),
            "n_peaks_med": np.full(n, 1.0),
            "temp_med_c": np.full(n, 25.0),
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_capacity_only_decline_reads_as_lli(self) -> None:
        """Capacity fades, peak voltage + height hold → LLI dominates."""
        n = 9
        monthly = self._monthly(anchored_cap_ah=np.linspace(50.0, 42.0, n))
        modes = attribute_modes(monthly)
        last = modes.iloc[-1]
        assert last["lli_frac"] > last["lam_frac"]

    def test_peak_collapse_reads_as_lam(self) -> None:
        """Peak height collapses and drifts → LAM dominates."""
        n = 9
        monthly = self._monthly(
            anchored_cap_ah=np.linspace(50.0, 46.0, n),
            main_peak_height=np.linspace(100.0, 60.0, n),
            main_peak_v=np.linspace(3.30, 3.35, n),
        )
        modes = attribute_modes(monthly)
        last = modes.iloc[-1]
        assert last["lam_frac"] > last["lli_frac"]

    def test_too_few_months_no_attribution(self) -> None:
        monthly = self._monthly().head(4)
        modes = attribute_modes(monthly)
        assert modes["lli_frac"].isna().all()


class TestMonthlySignaturesEmpty:

    def test_empty_frame_returns_empty_signatures(self) -> None:
        empty = pd.DataFrame(
            columns=["timestamp", "voltage_v", "current_a", "temperature_c"]
        )
        out = monthly_signatures(empty, capacity_ah=100.0, cells_series=16)
        assert out.empty
