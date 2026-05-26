"""Unit tests for the daily-KPI confidence gate on RTE.

The CASE expression in
``src/bess_fleet/pipeline/build_daily_kpis.py`` returns NULL unless
FOUR conditions all hold:

* ``energy_in_kwh   >= 0.10 × capacity_kwh``     (≥10 % nameplate
                                                  charged — not trickle)
* ``energy_out_kwh  >= 0.05 × capacity_kwh``     (≥5  % nameplate
                                                  discharged)
* ``energy_out / energy_in <= 1.05``              (physically plausible)
* ``|soc_end - soc_start| <= 10``                 (cycle closes — day
                                                  forms a loop)

Charge / discharge thresholds are capacity-relative so the rule
generalises across fleet sizes (5 kWh residential → 1 MWh utility)
without re-tuning. The cycle-closure rule catches the subtle case where
energy totals look healthy but SoC drifted across the day boundary.

These tests run the actual CASE expression via DuckDB in-memory so the
SQL is validated end-to-end rather than reimplemented.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

# Default nameplate used in fixtures: matches the Figgener 8 kWh group
# (ID14/16/17/18). Override on a per-test basis where needed.
DEFAULT_CAPACITY_KWH = 8.0


# The CASE expression — copied verbatim from build_daily_kpis.py.
# Kept in sync with the production query; if the rule changes, this
# query must change too.
RTE_GATE_SQL = """
    SELECT
        label,
        capacity_kwh,
        energy_in_kwh,
        energy_out_kwh,
        soc_start,
        soc_end,
        CASE
            WHEN energy_in_kwh  >= capacity_kwh * 0.10
             AND energy_out_kwh >= capacity_kwh * 0.05
             AND energy_out_kwh / energy_in_kwh <= 1.05
             AND ABS(soc_end - soc_start) <= 10.0
            THEN energy_out_kwh / energy_in_kwh
            ELSE NULL
        END AS rte
    FROM daily
    ORDER BY label
"""


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB instance with a tiny 'daily' table.

    Each row pairs a human-readable label with a system's nameplate
    capacity plus daily (in, out, soc_start, soc_end). The label lets
    us assert against specific rows by name."""
    c = duckdb.connect(":memory:")
    c.sql("""
        CREATE TABLE daily (
            label           VARCHAR,
            capacity_kwh    DOUBLE,
            energy_in_kwh   DOUBLE,
            energy_out_kwh  DOUBLE,
            soc_start       DOUBLE,
            soc_end         DOUBLE
        )
    """)
    yield c
    c.close()


def _insert(
    con: duckdb.DuckDBPyConnection,
    label: str,
    e_in: float,
    e_out: float,
    soc_start: float = 50.0,
    soc_end: float = 50.0,
    capacity_kwh: float = DEFAULT_CAPACITY_KWH,
) -> None:
    """Insert one synthetic daily row. SoC defaults form a closed
    cycle so existing energy-gate tests aren't affected by the SoC
    condition. Capacity defaults to 8 kWh (Figgener residential)."""
    con.execute(
        "INSERT INTO daily VALUES (?, ?, ?, ?, ?, ?)",
        [label, capacity_kwh, e_in, e_out, soc_start, soc_end],
    )


def _get(df: pd.DataFrame, label: str) -> dict[str, float | None]:
    row = df[df["label"] == label].iloc[0]
    return {
        "rte": None if pd.isna(row["rte"]) else float(row["rte"]),
    }


# ─── Healthy / passing cases (cycle closes) ─────────────────────────


class TestHealthyDays:
    """Days where energy totals are reasonable AND the cycle closes."""

    def test_normal_day_yields_rte(self, con) -> None:
        """A typical 8 kWh residential day: ~5 kWh in, ~4 kWh out
        (~80 % RTE), SoC returns to start."""
        _insert(con, "normal", e_in=5.0, e_out=4.0, soc_start=20.0, soc_end=22.0)
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "normal")["rte"] == pytest.approx(0.8)

    def test_high_efficiency_day_within_cap(self, con) -> None:
        """Ratio just below the 1.05 cap, cycle closes — should pass."""
        _insert(con, "near_cap", e_in=10.0, e_out=10.4, capacity_kwh=20.0)
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "near_cap")["rte"] == pytest.approx(1.04, abs=0.001)

    def test_exactly_at_capacity_minimums_passes(self, con) -> None:
        """Boundary: energy_in exactly 10 % of nameplate AND
        energy_out exactly 5 % — inclusive."""
        # 8 kWh × 0.10 = 0.80;  8 kWh × 0.05 = 0.40
        _insert(con, "boundary", e_in=0.80, e_out=0.40)
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "boundary")["rte"] == pytest.approx(0.5)

# ─── Confidence-gate failures (NULL outcomes) ───────────────────────


class TestEnergyGateFailures:
    """The capacity-relative energy thresholds and the ratio cap."""

    def test_insufficient_charging_nulled(self, con) -> None:
        """energy_in < 10 % of nameplate → NULL, even if discharge ok.
        For 8 kWh nameplate, threshold is 0.80 kWh."""
        _insert(con, "low_in", e_in=0.5, e_out=0.6)
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "low_in")["rte"] is None

    def test_insufficient_discharging_nulled(self, con) -> None:
        """energy_out < 5 % of nameplate → NULL.
        For 8 kWh nameplate, threshold is 0.40 kWh."""
        _insert(con, "low_out", e_in=8.0, e_out=0.3)
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "low_out")["rte"] is None

    def test_implausible_ratio_nulled(self, con) -> None:
        """ratio > 1.05 → NULL. Catches cross-day-boundary cycles
        where today's discharge draws on yesterday's stored charge."""
        _insert(con, "impossible", e_in=2.0, e_out=2.5)
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "impossible")["rte"] is None

    def test_just_above_ratio_cap_nulled(self, con) -> None:
        """Boundary: ratio of 1.06 must fail (cap is <= 1.05)."""
        _insert(con, "above_cap", e_in=10.0, e_out=10.6, capacity_kwh=20.0)
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "above_cap")["rte"] is None


# ─── Capacity-relative behaviour — the new generality ───────────────


class TestCapacityRelativeThresholds:
    """The same absolute energy values pass or fail depending on
    nameplate. This is exactly the generalisation the capacity-
    relative formulation buys you."""

    def test_same_energy_passes_small_pack_fails_big_pack(self, con) -> None:
        """0.85 kWh in is 10.6 % of 8 kWh (passes) but only 0.85 % of
        100 kWh (fails). Same absolute value, different verdict by
        rack size — exactly what we want."""
        _insert(con, "small_pack_8kwh",  e_in=0.85, e_out=0.50, capacity_kwh=8.0)
        _insert(con, "big_pack_100kwh", e_in=0.85, e_out=0.50, capacity_kwh=100.0)
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "small_pack_8kwh")["rte"] is not None
        assert _get(df, "big_pack_100kwh")["rte"] is None

    def test_utility_scale_pack_with_realistic_throughput(self, con) -> None:
        """1 MWh utility-scale rack: 200 kWh in, 170 kWh out.
        That's 20 % charge, 17 % discharge — both above thresholds.
        RTE = 0.85, cycle closes within 8 pp."""
        _insert(
            con, "utility",
            e_in=200.0, e_out=170.0,
            soc_start=30.0, soc_end=33.0,
            capacity_kwh=1000.0,
        )
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "utility")["rte"] == pytest.approx(0.85)

    @pytest.mark.parametrize("capacity,e_in,e_out,expected_pass", [
        # Same fractional thresholds, different rack sizes
        (5.0,    0.50,   0.25,  True),    # 10 % / 5 % exactly — passes
        (5.0,    0.49,   0.25,  False),   # just under 10 % charge → fails
        (5.0,    0.50,   0.24,  False),   # just under 5 % discharge → fails
        (50.0,   5.0,    2.5,   True),    # mid-scale, exactly at thresholds
        (50.0,   4.99,   2.5,   False),
        (500.0,  50.0,   25.0,  True),    # large utility, exactly at thresholds
        (500.0,  50.0,   24.99, False),
    ])
    def test_threshold_scales_with_capacity(
        self, con, capacity, e_in, e_out, expected_pass,
    ):
        """The capacity-relative thresholds behave identically across
        a 100× range of nameplate sizes."""
        _insert(
            con, f"cap_{capacity}_in_{e_in}_out_{e_out}",
            e_in=e_in, e_out=e_out, capacity_kwh=capacity,
        )
        df = con.sql(RTE_GATE_SQL).df()
        rte = _get(df, f"cap_{capacity}_in_{e_in}_out_{e_out}")["rte"]
        if expected_pass:
            assert rte is not None
        else:
            assert rte is None


# ─── SoC-closure condition (the fourth gate) ────────────────────────


class TestSocClosureGate:
    """The fourth condition: |soc_end - soc_start| <= 10 pp.

    Catches subtle cases where energy totals look fine but the cycle
    didn't close — the day either banked charge for tomorrow (SoC up)
    or drained stored charge (SoC down). In both cases the in/out
    ratio is not a true RTE.
    """

    def test_banked_charge_day_nulled(self, con) -> None:
        """20 % → 35 %: started low, charged a lot, discharged some,
        ended higher. Some charging went into raising SoC, not into
        producing the discharge. RTE understates true efficiency."""
        _insert(
            con, "banked",
            e_in=5.0, e_out=4.0,
            soc_start=20.0, soc_end=35.0,   # +15 pp drift
        )
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "banked")["rte"] is None

    def test_drained_storage_day_nulled(self, con) -> None:
        """80 % → 60 %: started high, drained stored charge. Even if
        the ratio passes (e.g. some charging happened), the day is
        a partial cycle and shouldn't contribute to daily RTE."""
        _insert(
            con, "drained",
            e_in=3.0, e_out=3.0,
            soc_start=80.0, soc_end=60.0,   # -20 pp drift
        )
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "drained")["rte"] is None

    def test_small_drift_passes(self, con) -> None:
        """Overnight self-discharge of 1-3 pp is normal — must pass."""
        _insert(
            con, "self_discharge",
            e_in=4.0, e_out=3.2,
            soc_start=55.0, soc_end=53.0,   # -2 pp normal drift
        )
        df = con.sql(RTE_GATE_SQL).df()
        assert _get(df, "self_discharge")["rte"] == pytest.approx(0.8)

    @pytest.mark.parametrize("delta_soc,expected_passes", [
        (0.0,   True),   # perfect closure
        (5.0,   True),   # well within tolerance
        (10.0,  True),   # exactly at boundary (inclusive)
        (10.1,  False),  # just over
        (15.0,  False),  # clearly partial cycle
        (-10.0, True),   # negative direction, still within
        (-15.0, False),  # negative direction, too much
    ])
    def test_closure_threshold_boundary(self, con, delta_soc, expected_passes) -> None:
        """The 10-pp tolerance is inclusive at the boundary, exclusive
        beyond. Works in both directions (SoC up or down)."""
        _insert(
            con, f"drift_{delta_soc}",
            e_in=5.0, e_out=4.0,
            soc_start=50.0, soc_end=50.0 + delta_soc,
        )
        df = con.sql(RTE_GATE_SQL).df()
        result = _get(df, f"drift_{delta_soc}")
        if expected_passes:
            assert result["rte"] is not None
        else:
            assert result["rte"] is None


# ─── Aggregate behaviour ────────────────────────────────────────────


class TestRealisticMixedFleet:
    """A small batch where days fail for different reasons. Mirrors
    the headline observation that ~30 % of daily rows return NULL once
    all four confidence gates are applied."""

    def test_mixed_batch_returns_expected_null_count(self, con) -> None:
        rows = [
            # label,           e_in, e_out, soc_start, soc_end  # reason
            ("healthy_1",       5.0,  4.0,  20.0,      22.0),   # pass
            ("healthy_2",       3.0,  2.5,  40.0,      41.0),   # pass
            ("low_in",          0.2,  0.3,  50.0,      50.0),   # fail (energy_in < 10 % of 8)
            ("low_out",         5.0,  0.1,  50.0,      50.0),   # fail (energy_out < 5 % of 8)
            ("impossible",      2.0,  3.0,  80.0,      50.0),   # fail (ratio + drift)
            ("healthy_3",       8.0,  7.0,  30.0,      32.0),   # pass
            ("banked_charge",   6.0,  4.0,  20.0,      40.0),   # fail (cycle didn't close)
            ("drained_storage", 2.0,  2.0,  75.0,      55.0),   # fail (cycle didn't close)
        ]
        for label, e_in, e_out, soc_s, soc_e in rows:
            _insert(con, label, e_in, e_out, soc_s, soc_e)
        df = con.sql(RTE_GATE_SQL).df()
        assert df["rte"].notna().sum() == 3
        assert df["rte"].isna().sum() == 5
