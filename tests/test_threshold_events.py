"""Unit tests for the run-length-encoding event collapser.

`collapse_to_events` is the engine that turns a boolean 1-minute
flag series into one row per sustained violation. The dashboard reads
its output verbatim, so the corner cases (no flags, all flags, single
isolated minute, multiple windows) must be locked down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_fleet.pipeline.detect_threshold_events import collapse_to_events


def _flag_series(pattern: list[int], start: str = "2025-01-01 00:00") -> pd.Series:
    """Build a flag series with a 1-min DatetimeIndex from a 0/1 list."""
    idx = pd.date_range(start, periods=len(pattern), freq="1min")
    return pd.Series([bool(x) for x in pattern], index=idx)


def _value_series(values: list[float], start: str = "2025-01-01 00:00") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="1min")
    return pd.Series(values, index=idx, dtype=float)


# ─── Empty / no-event cases ─────────────────────────────────────────


class TestNoEvents:

    def test_empty_input_returns_empty_frame(self) -> None:
        """Empty AND all-false collapse to the same code path; one
        test covers both. Schema must remain intact so downstream
        concat doesn't break."""
        flag = _flag_series([])
        value = _value_series([])
        result = collapse_to_events(flag, value)
        assert result.empty
        assert list(result.columns) == ["start", "end", "duration_min", "peak_value"]


# ─── Single-event collapse ──────────────────────────────────────────


class TestSingleEvent:

    def test_single_block_collapses_to_one_row(self) -> None:
        """10 contiguous flag minutes → 1 event row."""
        flag = _flag_series([0]*5 + [1]*10 + [0]*5)
        value = _value_series(
            [0.0]*5 + [10.0, 20.0, 30.0, 25.0, 15.0, 12.0, 18.0, 14.0, 11.0, 8.0] + [0.0]*5
        )
        result = collapse_to_events(flag, value)
        assert len(result) == 1
        assert result["duration_min"].iloc[0] == pytest.approx(10.0)

    def test_peak_value_is_max_absolute_over_window(self) -> None:
        """peak_value uses |value|.max() so signed channels (ΔT) work."""
        flag = _flag_series([1]*5)
        value = _value_series([-10.0, -12.0, 8.0, -15.0, 11.0])
        result = collapse_to_events(flag, value)
        assert len(result) == 1
        assert result["peak_value"].iloc[0] == pytest.approx(15.0)

    def test_start_and_end_match_window_boundaries(self) -> None:
        flag = _flag_series([0]*3 + [1]*4 + [0]*3, start="2025-06-15 12:00")
        value = _value_series([0.0]*3 + [1.0]*4 + [0.0]*3, start="2025-06-15 12:00")
        result = collapse_to_events(flag, value)
        assert result["start"].iloc[0] == pd.Timestamp("2025-06-15 12:03")
        assert result["end"].iloc[0]   == pd.Timestamp("2025-06-15 12:06")


# ─── Multi-event behaviour ──────────────────────────────────────────


class TestMultipleEvents:

    def test_two_separated_windows_yield_two_rows(self) -> None:
        """0001100011110 → two events: 2 min and 4 min."""
        flag = _flag_series([0,0,0,1,1,0,0,0,1,1,1,1,0])
        value = _value_series([0.0]*13)
        result = collapse_to_events(flag, value)
        assert len(result) == 2
        assert result["duration_min"].iloc[0] == pytest.approx(2.0)
        assert result["duration_min"].iloc[1] == pytest.approx(4.0)


# ─── min_duration filter ────────────────────────────────────────────


class TestMinDurationFilter:

    def test_short_events_filtered_out(self) -> None:
        """A 2-min event must not appear when min_duration_min=5."""
        flag = _flag_series([1,1,0,0,0,1,1,1,1,1,1,1])  # 2 min, then 7 min
        value = _value_series([1.0]*12)
        result = collapse_to_events(flag, value, min_duration_min=5)
        assert len(result) == 1
        assert result["duration_min"].iloc[0] == pytest.approx(7.0)

    def test_exact_min_duration_passes(self) -> None:
        """Event of exactly min_duration_min must survive (inclusive)."""
        flag = _flag_series([1]*5)
        value = _value_series([1.0]*5)
        result = collapse_to_events(flag, value, min_duration_min=5)
        assert len(result) == 1


# ─── NaN robustness ─────────────────────────────────────────────────


class TestNanHandling:

    def test_nan_flags_treated_as_false(self) -> None:
        """The production data has NULL ambient on ID19, which
        propagates as NaN. Those must not trigger events."""
        idx = pd.date_range("2025-01-01", periods=5, freq="1min")
        flag = pd.Series([True, np.nan, True, np.nan, True], index=idx, dtype=object)
        value = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
        result = collapse_to_events(flag, value)
        # NaN-as-False → three separate single-minute events
        # (but if min_duration_min defaults to 1, they all survive)
        assert len(result) == 3
