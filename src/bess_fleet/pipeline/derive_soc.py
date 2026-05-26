"""OCV-corrected coulomb-counted SoC for the LFP fleet.

Adds ``soc_pct`` (and a ``is_soc_anchor`` boolean) to every parquet in
``data/processed/`` in place. Pure hybrid SoC estimation — voltage-anchored
at relaxed-cell rest periods, current-integrated between anchors, drift
corrected by linear interpolation. The same algorithm production BMSs
use (Plett, *Battery Management Systems Vol 1*, ch. 3) — coulomb
counting alone integrates sensor offset forever; voltage alone is
useless in LFP's flat plateau.

Algorithm
---------
1. **Rest detection.** ``|power_kw| × 1000 < 50 W`` sustained for ≥30
   continuous min → the cell has relaxed; terminal voltage ≈ OCV.
2. **OCV anchor.** At each rest-period start, look up SoC from the
   literature LFP OCV curve (:data:`OCV_SOC_TABLE`).
3. **Coulomb count.** Unanchored cumulative ``Σ I·Δt / Q`` (Δt = 1/60 h,
   Q = ``capacity_ah`` from the identity table).
4. **Drift correction.** At every anchor, residual ``cc − anchor`` is
   the integrated drift since the last anchor. Linearly interpolate
   that residual between anchors, then subtract from cc. Forward/back-
   fill extends the first/last drift value past the boundary anchors.
5. **Clip** to ``[0, 100]``. Excursions outside are expected for a small
   fraction of samples — see notebook section 8 / Test 2.

Derived columns (float32 / bool, both nullable)
-----------------------------------------------
* ``soc_pct``        — final clipped SoC
* ``is_soc_anchor``  — ``True`` at OCV-anchor timestamps (useful for
                       overlaying anchor markers in dashboard charts)

Algorithm caveat — audited in ``notebooks/daily_kpis_eda.ipynb`` § 8
------------------------------------------------------------------
Eight stress tests confirm the implementation is mathematically sound:

* Anchor self-consistency RMSE = 0.0000 (by construction)
* Dispatch monotonicity ≥ 98.6 % across all systems
* Daily DoD vs |throughput Ah| Pearson r² 0.70–0.84 fleet-wide

The *reliability* story splits the fleet:

* **ID14, ID16, ID17, ID18** — anchors distribute well across the OCV
  curve (25–40 % in plateau, rest on the reliable cliffs); cross-anchor
  drift median <1 % SoC; pre-clip excursions <2 % of samples.
  **Absolute SoC trustworthy.**
* **ID19, ID20** — >90 % of anchors land in the LFP plateau (3.10–
  3.36 V, where the OCV→SoC lookup is ±5–10 % noisy by construction),
  and 15–20 months show zero anchors so SoC is pure-CC extrapolation
  in those windows. **Absolute SoC carries ±5 % uncertainty; relative
  metrics (DoD, trends, anomaly z-scores) remain trustworthy.**

This is a data limitation, not an algorithm bug — those two racks never
deeply discharge or fully charge, so OCV anchors are inherently noisy.

Run order
---------
    1. python -m scripts.lfp_to_1min_parquet      # raw zip → 1-min parquet
    2. python -m scripts.clean_temperatures       # sentinel scrub → processed/
    3. python -m scripts.load_identity            # XLSX → identity.parquet
    4. python -m scripts.derive_features          # ΔT, mode, energy, c_rate
    5. python -m scripts.derive_soc               # ← this script
    6. python -m scripts.build_daily_kpis         # (system × day) curated
    7. python -m scripts.detect_threshold_events  # threshold-event audit

Idempotent — re-run any step to refresh without manual cleanup. This
script drops any prior ``soc_pct`` / ``is_soc_anchor`` columns before
recomputing, so it's safe to re-run after tweaking thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_fleet.db import DATA_DIR
from bess_fleet.io import safe_to_parquet

PROCESSED_DIR = DATA_DIR / "processed"
IDENTITY_PATH = DATA_DIR / "identity.parquet"

# Literature LFP open-circuit voltage curve. The plateau between SoC
# 20–80 % is what makes voltage-alone SoC useless on this chemistry —
# 60 % of the operating range is compressed into 60 mV of cell voltage.
# Coulomb counting fills that gap; this lookup just anchors it.
OCV_SOC_TABLE = np.array([
    [0,   2.500], [1,   2.950], [5,   3.100], [10, 3.200], [20, 3.270],
    [30,  3.300], [50,  3.310], [70,  3.320], [80,  3.330], [90, 3.360],
    [95,  3.440], [99,  3.550], [100, 3.650],
])

REST_THRESHOLD_W  = 50    # |P| < 50 W → effectively idle on a ~8 kWh system
REST_DURATION_MIN = 30    # idle this long → cell relaxed → V ≈ OCV
DT_HOURS          = 1.0 / 60.0


def ocv_to_soc(cell_voltage_v: np.ndarray) -> np.ndarray:
    """LFP-cell OCV [V] → SoC [%] via 1-D linear interpolation.

    Saturates flat outside the table range — values below 2.50 V map
    to 0 %, above 3.65 V to 100 %. That's the right behaviour for a
    relaxed cell: a reading outside the OCV curve almost always means
    we hit an edge of the lookup, not that the cell is over/under.
    """
    return np.interp(cell_voltage_v, OCV_SOC_TABLE[:, 1], OCV_SOC_TABLE[:, 0])


def derive_soc(
    df: pd.DataFrame,
    capacity_ah: float,
    cells_series: int,
) -> pd.DataFrame:
    """OCV-corrected SoC for one system's cleaned 1-min frame.

    Parameters
    ----------
    df
        Frame with a ``timestamp`` column (any dtype convertible to
        ``datetime64``) sorted ascending, plus ``voltage_v``,
        ``current_a``, ``power_kw``.
    capacity_ah
        Pack capacity in amp-hours, from the identity table.
    cells_series
        Number of cells in series — used to derive cell voltage from
        pack voltage for the OCV lookup.

    Returns
    -------
    A copy of ``df`` with two new columns:

    * ``soc_pct`` — float32, clipped to [0, 100]
    * ``is_soc_anchor`` — bool, True at each rest-period start

    The function is pure: same input → same output, no I/O.
    """
    # Operate against a DatetimeIndex view so rolling("30min") is
    # time-aware (handles missing minutes correctly, unlike a fixed-
    # row window). Results get written back to the original frame
    # at the end so the caller's column order is preserved.
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    work = df.set_index(ts)
    cell_v = work["voltage_v"] / cells_series

    # ─ Step 1: rest detection ──────────────────────────────────────────
    is_idle = (work["power_kw"].abs() * 1000) < REST_THRESHOLD_W
    rested = (
        is_idle
        .rolling(f"{REST_DURATION_MIN}min", min_periods=20)
        .min()
        .fillna(0)
        .astype(bool)
    )
    # Rising edge of `rested` → the moment the cell first qualifies
    # as relaxed. Falling edge is the moment dispatch resumes.
    is_anchor = rested & ~rested.shift(fill_value=False)

    # ─ Step 2: OCV-anchor lookups ──────────────────────────────────────
    anchor_soc = pd.Series(np.nan, index=work.index)
    anchor_soc.loc[is_anchor] = ocv_to_soc(cell_v.loc[is_anchor].to_numpy())

    # ─ Step 3: coulomb counting (unanchored cumulative integral) ───────
    delta_soc_pct = work["current_a"] * DT_HOURS / capacity_ah * 100.0
    cc_unanchored = delta_soc_pct.cumsum()

    # ─ Step 4: linear-interp drift correction ──────────────────────────
    # `drift_at_anchor` is the integration error at each anchor:
    # how much the coulomb-counted value has wandered away from the
    # voltage-based truth since the previous anchor.
    drift_at_anchor = cc_unanchored - anchor_soc
    drift_interp    = (
        drift_at_anchor
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )
    soc_raw = cc_unanchored - drift_interp

    # ─ Step 5: clip & write back to caller's frame layout ──────────────
    out = df.copy()
    out["soc_pct"]       = soc_raw.clip(0.0, 100.0).astype("float32").to_numpy()
    out["is_soc_anchor"] = is_anchor.to_numpy()
    return out


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"no parquets in {PROCESSED_DIR}. Run derive_features.py first."
        )
    if not IDENTITY_PATH.exists():
        raise SystemExit(
            f"missing {IDENTITY_PATH}. Run `python -m scripts.load_identity` first."
        )

    ident = pd.read_parquet(IDENTITY_PATH)
    cap_lookup   = dict(zip(ident["system_id"], ident["capacity_ah"],   strict=True))
    cells_lookup = dict(zip(ident["system_id"], ident["cells_series"],  strict=True))

    print(
        f"deriving SoC for {len(files)} parquets in {PROCESSED_DIR}\n",
        flush=True,
    )

    for path in files:
        sid = path.stem
        if sid not in cap_lookup:
            print(f"  [{sid}] SKIP — no identity row", flush=True)
            continue

        df = pd.read_parquet(path)
        # Drop any prior derivation so re-runs are clean
        df = df.drop(columns=["soc_pct", "is_soc_anchor"], errors="ignore")

        out = derive_soc(
            df,
            capacity_ah=float(cap_lookup[sid]),
            cells_series=int(cells_lookup[sid]),
        )
        safe_to_parquet(out, path, index=False, compression="snappy")

        soc = out["soc_pct"]
        n_anch = int(out["is_soc_anchor"].sum())
        print(
            f"  [{sid}] anchors={n_anch:>5,}  "
            f"SoC μ={soc.mean():5.1f}%  median={soc.median():5.1f}%  "
            f"range=[{soc.min():5.1f}, {soc.max():5.1f}]  "
            f"rows={len(out):,}",
            flush=True,
        )

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
