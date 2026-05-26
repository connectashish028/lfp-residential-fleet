"""Build per-(system, day) KPI table from the cleaned 1-min telemetry.

Reads ``telemetry_1min_clean`` joined with ``identity`` (DuckDB views),
aggregates by ``(system_id, date)``, and writes
``data/curated/daily_kpis.parquet``. One row per system per day.

KPI schema:

* ``system_id``, ``date``                  — composite key
* ``n_samples``                            — count of 1-min rows present
* ``coverage_pct``                         — n_samples / 1440 × 100
                                             (low-confidence < 80 %)
* ``energy_in_kwh``, ``energy_out_kwh``    — daily charge / discharge totals
* ``throughput_kwh``                       — sum of |energy_kwh_step|
* ``rte``                                  — energy_out / energy_in
                                             (NULL when no charging)
* ``efc``                                  — throughput / (2 × capacity_kwh)
* ``idle_fraction``                        — AVG(is_idle)
* ``mean_dt_c``, ``max_dt_c``              — daily thermal residual
                                             (NaN-aware — ID19 carries NULLs)
* ``mean_c_rate``, ``max_c_rate``          — daily duty intensity

Decisions worth flagging:

1. **Daily RTE is gated by four confidence rules.** Residential cycles
   cross day boundaries, so the calendar-day ratio breaks down whenever
   the battery doesn't return to the same SoC by midnight. We return
   NULL unless ALL four conditions hold:

     * ``energy_in_kwh  >= 0.10 × capacity_kwh``  (≥10 % of nameplate
                                                   charged — not trickle)
     * ``energy_out_kwh >= 0.05 × capacity_kwh``  (≥5 % of nameplate
                                                   discharged)
     * raw ratio ``<= 1.05``                       (physically plausible)
     * ``|soc_end - soc_start| <= 10 pp``          (cycle closes)

   Charge and discharge thresholds are **fractions of nameplate
   capacity**, not absolute kWh, so the same rule generalises across
   fleet sizes (5 kWh residential to 1 MWh utility) without changing
   thresholds. The fractions are calibrated against typical residential
   daily-energy distributions: ≥10 % rules out trickle-only days,
   ≥5 % rules out solar-fill-without-load days.

   The fourth condition is the subtle one. ``energy_out / energy_in``
   only equals true RTE when the battery starts and ends the day at
   the same SoC. If the day banked energy (SoC went up) the ratio
   understates RTE; if the day drained stored charge (SoC went down)
   the ratio overstates RTE — and may overshoot the 1.05 cap. Requiring
   day-start and day-end SoC to be within 10 pp of each other filters
   out partial cycles where the day's totals don't form a closed loop.

   This leaves the noisy fringe nulled out — typical days where the
   cycle closes survive, partial-cycle days get filtered. Monthly RTE
   is the headline; daily is for trend / event detection where coverage
   and discipline matter.
2. ``coverage_pct`` uses a 1440-minute expected baseline. DST transition
   days have 1380 or 1500 expected minutes (~2 days a year per
   timezone); not worth correcting at this granularity.
3. NaN in ``thermal_delta_c`` (ID19's broken ambient) propagates through
   ``AVG`` / ``MAX`` via DuckDB's native skip-NULL behaviour. Days where
   every ambient sample is missing yield NULL ``mean_dt_c``.

Run with::

    python -m bess_fleet.pipeline.build_daily_kpis
"""

from __future__ import annotations

from bess_fleet.db import DATA_DIR, connect
from bess_fleet.io import safe_to_parquet

OUT_PATH = DATA_DIR / "curated" / "daily_kpis.parquet"

# ─── RTE confidence-gate parameters ────────────────────────────────────
# All thresholds are dimensionless / capacity-relative so the same rule
# generalises across fleet sizes without re-tuning.

# Minimum charge for a day to count: 10 % of nameplate kWh. Rules out
# trickle-only days where the day's energy is sensor noise.
RTE_MIN_CHARGE_FRAC: float = 0.10

# Minimum discharge for a day to count: 5 % of nameplate kWh. Rules out
# pure solar-fill days where the load drew nothing from the battery.
RTE_MIN_DISCHARGE_FRAC: float = 0.05

# Physical-plausibility cap on the in/out ratio. Anything above this
# means more energy came out than went in, i.e. cross-day-boundary leak
# from a charge banked the previous day.
RTE_RATIO_CAP: float = 1.05

# Tolerance for the SoC-closure rule on daily RTE. Residential overnight
# self-discharge is typically 1-3 pp; 10 pp leaves comfortable margin for
# normal operation while still excluding partial-cycle days where one
# day's totals don't form a closed loop.
SOC_CLOSURE_TOLERANCE_PP: float = 10.0


DAILY_KPI_SQL = f"""
SELECT
    t.system_id,
    date_trunc('day', t.timestamp)::DATE                       AS date,

    COUNT(*)                                                    AS n_samples,
    COUNT(*) / 1440.0 * 100.0                                   AS coverage_pct,

    SUM(t.energy_in_kwh_step)                                   AS energy_in_kwh,
    SUM(t.energy_out_kwh_step)                                  AS energy_out_kwh,
    SUM(ABS(t.energy_kwh_step))                                 AS throughput_kwh,

    -- Day-boundary SoC for the cycle-closure check. arg_min/arg_max
    -- return the soc_pct at the row with the min/max timestamp, i.e.
    -- the day's first and last sample.
    arg_min(t.soc_pct, t.timestamp)                              AS soc_start,
    arg_max(t.soc_pct, t.timestamp)                              AS soc_end,

    -- Daily RTE with four-condition confidence gating — see module
    -- docstring. Charge / discharge thresholds are capacity-relative
    -- (fractions of nameplate kWh) so the rule generalises across
    -- fleet sizes. MAX(i.capacity_kwh) is just a constant lookup since
    -- capacity is uniform per system_id within the join.
    CASE
        WHEN SUM(t.energy_in_kwh_step)
                >= MAX(i.capacity_kwh) * {RTE_MIN_CHARGE_FRAC}
         AND SUM(t.energy_out_kwh_step)
                >= MAX(i.capacity_kwh) * {RTE_MIN_DISCHARGE_FRAC}
         AND SUM(t.energy_out_kwh_step) / SUM(t.energy_in_kwh_step)
                <= {RTE_RATIO_CAP}
         AND ABS(
               arg_max(t.soc_pct, t.timestamp)
             - arg_min(t.soc_pct, t.timestamp)
           ) <= {SOC_CLOSURE_TOLERANCE_PP}
        THEN SUM(t.energy_out_kwh_step) / SUM(t.energy_in_kwh_step)
        ELSE NULL
    END                                                          AS rte,

    -- EFC: throughput / (2 × capacity). MAX(i.capacity_kwh) is just a
    -- constant lookup since capacity_kwh is the same for every row of a
    -- given system_id (one-to-many join with identity).
    SUM(ABS(t.energy_kwh_step)) / (2.0 * MAX(i.capacity_kwh))    AS efc,

    AVG(CAST(t.is_idle AS DOUBLE))                               AS idle_fraction,

    -- Thermal — NaN-aware. Days where every ambient is NULL yield NULL.
    AVG(t.thermal_delta_c)                                       AS mean_dt_c,
    MAX(t.thermal_delta_c)                                       AS max_dt_c,

    AVG(t.c_rate)                                                AS mean_c_rate,
    MAX(t.c_rate)                                                AS max_c_rate
FROM telemetry_1min_clean t
JOIN identity i USING (system_id)
GROUP BY t.system_id, date
ORDER BY t.system_id, date
"""


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        df = con.sql(DAILY_KPI_SQL).df()

    safe_to_parquet(df, OUT_PATH, index=False, compression="snappy")
    print(
        f"wrote {OUT_PATH}: {len(df):,} (system × day) rows, "
        f"{OUT_PATH.stat().st_size / 1e6:.2f} MB\n",
        flush=True,
    )

    # Per-system summary as a quick sanity check
    print("Per-system summary (daily KPI table):")
    summary = (
        df.groupby("system_id")
        .agg(
            n_days=("date", "size"),
            first_day=("date", "min"),
            last_day=("date", "max"),
            mean_rte=("rte", "mean"),
            mean_idle=("idle_fraction", "mean"),
            mean_dt=("mean_dt_c", "mean"),
            mean_c=("mean_c_rate", "mean"),
            n_rte_null=("rte", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
