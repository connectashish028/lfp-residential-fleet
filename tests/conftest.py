"""Pytest configuration for the BESS Fleet Health test suite.

All pipeline modules live under ``bess_fleet.pipeline`` and are
importable via the editable install — no ``sys.path`` munging needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ─── Shared fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def synth_telemetry_resting() -> pd.DataFrame:
    """120 minutes of 1-min telemetry at full rest.

    Constant cell voltage in the LFP plateau, zero current, zero power.
    Long enough (≥ 30 min idle) to trigger one SoC anchor.
    """
    n = 120
    ts = pd.date_range("2025-01-01 00:00", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "timestamp":   ts,
            "voltage_v":   np.full(n, 51.0, dtype=float),    # cell ≈ 3.1875 V
            "current_a":   np.zeros(n, dtype=float),
            "power_kw":    np.zeros(n, dtype=float),
        }
    )


@pytest.fixture()
def synth_telemetry_charging() -> pd.DataFrame:
    """30 min rest, then 60 min steady charging at +5 A.

    Used to assert SoC increases monotonically once the rest anchor
    fires and current starts flowing.
    """
    n_rest, n_chrg = 30, 60
    n = n_rest + n_chrg
    ts = pd.date_range("2025-01-01 00:00", periods=n, freq="1min")
    current = np.concatenate([np.zeros(n_rest), np.full(n_chrg, 5.0)])
    voltage = np.full(n, 51.0)
    # power kw = V × I / 1000; sign matches current direction (charging +)
    power = voltage * current / 1000.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "voltage_v": voltage,
            "current_a": current,
            "power_kw":  power,
        }
    )


@pytest.fixture()
def synth_telemetry_discharging() -> pd.DataFrame:
    """30 min rest, then 60 min steady discharge at -5 A."""
    n_rest, n_dis = 30, 60
    n = n_rest + n_dis
    ts = pd.date_range("2025-01-01 00:00", periods=n, freq="1min")
    current = np.concatenate([np.zeros(n_rest), np.full(n_dis, -5.0)])
    voltage = np.full(n, 51.0)
    power = voltage * current / 1000.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "voltage_v": voltage,
            "current_a": current,
            "power_kw":  power,
        }
    )
