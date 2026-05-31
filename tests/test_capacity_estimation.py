"""Unit tests for the Figgener capacity-estimation core (eq 1–3).

Pin the pure functions against synthetic signals with known answers: the
ECM must recover a known V_OCV, the OCV→SOC slope must carry the right
sensitivity, the offset must zero a closed cycle, the capacity must fall
out of a known charge / swing, and the weighted ageing fit must recover a
known fade rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_fleet.pipeline.capacity_estimation import (
    Rest,
    estimate_capacity,
    estimate_offset_current,
    fit_relaxation_ocv,
    ocv_to_soc_with_slope,
    weighted_ageing_fit,
)


class TestEcmFit:
    def _relaxation(self, v_ocv: float, sign: int, dur_s: int = 1800) -> tuple:
        rng = np.random.default_rng(1)
        t = np.arange(0, dur_s, 1.0)
        v = v_ocv + sign * 0.12 * np.exp(-t / 25.0) + sign * 0.06 * np.exp(-t / 700.0)
        return t, v + rng.normal(0, 2e-3, t.size)

    def test_recovers_known_ocv(self) -> None:
        for v_ocv, sign in [(4.00, -1), (3.20, 1)]:
            t, v = self._relaxation(v_ocv, sign)
            out = fit_relaxation_ocv(t, v)
            assert out is not None
            assert out[0] == pytest.approx(v_ocv, abs=2e-3)   # within 2 mV
            assert out[1] < 0.01                              # tight σ

    def test_too_short_returns_none(self) -> None:
        t = np.arange(0, 120, 1.0)               # 2 min — below the span guard
        v = 3.3 - 0.05 * np.exp(-t / 30.0)
        assert fit_relaxation_ocv(t, v) is None


class TestOcvSlope:
    def test_flat_plateau_has_larger_slope_than_sloped(self) -> None:
        """At a flat LFP plateau voltage the dSOC/dOCV is far larger than
        on a sloped NMC curve — the same σ_OCV → a far larger σ_SOC."""
        _, slope_lfp = ocv_to_soc_with_slope(3.30, "LFP")
        _, slope_nmc = ocv_to_soc_with_slope(3.85, "NMC")
        assert slope_lfp > 3 * slope_nmc


class TestOffsetCurrent:
    def test_closed_cycle_recovers_offset(self) -> None:
        """Two same-kind rests 10 h apart whose coulomb count drifted by
        −1 Ah imply an offset of −0.1 A."""
        t0 = pd.Timestamp("2020-01-01")
        rests = [
            Rest(t0, t0, "top", cum_ah_end=0.0, cell_v_1min=4.0),
            Rest(t0 + pd.Timedelta(hours=10), t0 + pd.Timedelta(hours=10),
                 "top", cum_ah_end=-1.0, cell_v_1min=4.0),
        ]
        i_off, _ = estimate_offset_current(rests)
        assert i_off == pytest.approx(-0.1, abs=1e-3)


class TestCapacityEstimate:
    def test_recovers_known_capacity(self) -> None:
        """A top→bottom discharge moving 12 Ah across an 80 % SOC swing on
        a 15 Ah pack → C_usable = 15 Ah → SOH 100 %."""
        t0 = pd.Timestamp("2020-06-01 12:00")
        t1 = t0 + pd.Timedelta(hours=8)
        # NMC OCVs chosen so SOC_top≈0.93, SOC_bottom≈0.13 → swing≈0.80.
        top = Rest(t0, t0, "top", cum_ah_end=12.0, cell_v_1min=4.1)
        bot = Rest(t1, t1, "bottom", cum_ah_end=0.0, cell_v_1min=3.5)
        top.ocv, top.sigma_ocv = 4.14, 0.002
        bot.ocv, bot.sigma_ocv = 3.55, 0.002
        ests = estimate_capacity([top, bot], "NMC", 15.0, i_offset=0.0, sigma_offset=0.0)
        assert len(ests) == 1
        # swing isn't exactly 0.8 (literature OCV), so allow a band.
        assert 12.0 <= ests[0]["capacity_ah"] <= 18.0
        assert ests[0]["sigma_soh_pp"] > 0


class TestAgeingFit:
    def test_recovers_known_fade_rate(self) -> None:
        n = 24
        months = pd.date_range("2018-01-01", periods=n, freq="MS")
        # 3 pp/yr fade over 2 years from 100 %.
        soh = 100.0 - 3.0 * (np.arange(n) / 12.0)
        est = pd.DataFrame({
            "timestamp": months, "soh_pct": soh,
            "sigma_soh_pp": np.full(n, 1.0),
        })
        out = weighted_ageing_fit(est, "IDxx", "NMC")
        assert out["ageing_pct_per_yr"] == pytest.approx(3.0, abs=0.1)
        assert out["n_estimates"] == n

    def test_too_few_months_no_rate(self) -> None:
        months = pd.date_range("2020-01-01", periods=3, freq="MS")
        est = pd.DataFrame({
            "timestamp": months, "soh_pct": [100.0, 99.0, 98.0],
            "sigma_soh_pp": [1.0, 1.0, 1.0],
        })
        out = weighted_ageing_fit(est, "IDxx", "LFP")
        assert np.isnan(out["ageing_pct_per_yr"])
