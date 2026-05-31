"""Degradation-mode estimation from field data — ICA / DVA on
reconstructed quasi-OCV sweeps.

This is the diagnostic that distinguishes *which* ageing mechanism is
consuming a cell's capacity, not merely *how much* is gone. It mirrors
the method of Figgener et al. (*Nature Energy* 2024,
``10.1038/s41560-024-01620-9``) and the degradation-mode follow-up
(arXiv ``2411.08025``): reconstruct quasi-open-circuit-voltage curves
from low-dynamic field operation, then read the electrode signatures off
their derivatives.

Why this matters across chemistries
-----------------------------------
A flat-OCV **LFP** cell hides its state of charge inside ~60 mV, so its
incremental-capacity curve is one tall, narrow spike — exquisitely
sensitive but information-poor, and its capacity is hard to pin from a
partial field sweep because tiny voltage errors map to large charge
errors. A sloped **NMC / LMO-NMC** cell spreads SoC across ~1.2 V, so its
ICA curve carries resolvable peaks and its capacity anchors cleanly. The
*same* code therefore yields very different *confidence* per chemistry,
and quantifying that gap is the point of running it cross-chemistry — it
is exactly the kind of validation a state estimator needs before it goes
near a product BMS.

Method
------
1. **Sweep extraction.** A usable quasi-OCV sweep is a contiguous,
   same-sign, low-C-rate (|I| ≤ :data:`SWEEP_MAX_CRATE`) charge or
   discharge that covers ≥ :data:`SWEEP_MIN_DSOC` of capacity with steady
   current. Residential systems produce these nightly (slow household
   supply) and on gentle solar charges. At low C-rate the IR
   overpotential is small and roughly constant, so terminal voltage
   tracks OCV up to an offset.
2. **Voltage-anchored capacity.** Following the field method, capacity is
   the charge moved between two *fixed* cell-voltage anchors
   (:data:`CHEM_ANCHORS`). Measuring over the same voltage window every
   time makes months comparable — unlike a raw sweep span, which just
   reflects how deep that night's discharge happened to go.
3. **ICA / DVA.** Within a sweep, pair capacity moved ``q`` [Ah] with
   cell voltage ``V``, resample, smooth (Savitzky-Golay — field data is
   noisy), and differentiate: ``ICA = dq/dV`` (peaks → phase
   transitions), ``DVA = dV/dq``.
4. **Mode attribution.** Decompose the change over time into **LLI**
   (loss of lithium inventory — capacity falls while the ICA peak voltage
   holds: signatures *translate*) vs **LAM** (loss of active material —
   the peak *shrinks and drifts*: the curve's *shape* changes). Reported
   as a fraction with an explicit per-chemistry confidence, because field
   sweeps are warmer, faster and noisier than a lab C/30 reference. These
   are *operational* degradation indicators — which mechanism dominates
   and when it accelerated — not teardown-grade numbers.

Output: ``data/curated/degradation_modes.parquet`` — one row per
(system, month) with the signature features and the per-month mode split.

Run with::

    python -m bess_fleet.pipeline.degradation_modes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from bess_fleet.db import DATA_DIR
from bess_fleet.io import safe_to_parquet

BRONZE_DIR = DATA_DIR / "lfp_1min"
IDENTITY_PATH = DATA_DIR / "identity.parquet"
OUT_PATH = DATA_DIR / "curated" / "degradation_modes.parquet"

# ─── Sweep-extraction parameters ───────────────────────────────────────────
SWEEP_MIN_CRATE = 0.01    # below this the cell is essentially idle
SWEEP_MAX_CRATE = 0.20    # above this IR overpotential corrupts the OCV proxy
SWEEP_MIN_DSOC  = 0.20    # a sweep must cover ≥20 % SoC to resolve peaks
SWEEP_GAP_MIN   = 4.0     # a time gap >4 min ends a sweep (missing data)
SWEEP_MIN_ROWS  = 40      # need enough points to smooth + differentiate

# Fixed cell-voltage anchors per chemistry — capacity is the charge moved
# between them. Chosen inside the band a typical overnight sweep
# traverses, so most sweeps yield a comparable measurement. The LFP pair
# sits in the plateau (large Ah, but voltage-noise-sensitive — that
# sensitivity is the headline cross-chemistry result, reported as a
# confidence metric rather than hidden).
CHEM_ANCHORS: dict[str, tuple[float, float]] = {
    "LFP": (3.240, 3.300),
    "NMC": (3.650, 3.950),
    "LMO": (3.600, 3.900),
}
_DEFAULT_ANCHORS = (3.300, 3.700)

# ─── ICA / DVA grid + smoothing ────────────────────────────────────────────
GRID_POINTS      = 200
SAVGOL_WINDOW    = 21
SAVGOL_POLYORDER = 3
PEAK_PROMINENCE_FRAC = 0.15   # prominence floor as a fraction of curve max
MAX_PEAKS        = 5
MAX_ICA_PER_MONTH = 40        # cap the (expensive) ICA work per month

# ─── Mode attribution ──────────────────────────────────────────────────────
BASELINE_MONTHS  = 3
MIN_MONTHS       = 6
SMOOTH_MONTHS    = 3          # rolling-median window on the capacity trend
PEAK_V_SHIFT_REF = 0.020      # peak drift beyond this (V) reads as LAM


@dataclass
class Sweep:
    """One low-dynamic quasi-OCV traversal, as numpy arrays (cheap)."""

    ts0: pd.Timestamp
    cell_v: np.ndarray[Any, Any]   # per-sample cell voltage
    q_ah: np.ndarray[Any, Any]     # Ah moved since sweep start (monotone ↑)
    temp_med: float


def reconstruct_sweeps(
    df: pd.DataFrame,
    capacity_ah: float,
    cells_series: int,
) -> list[Sweep]:
    """Segment a system's 1-min frame into low-dynamic quasi-OCV sweeps.

    Linear-time: segment boundaries are found once from the slow/same-sign
    mask, then each contiguous segment is sliced directly — no per-segment
    rescans.
    """
    if df.empty:
        return []
    work = df.sort_values("timestamp").reset_index(drop=True)
    ts = work["timestamp"].to_numpy()
    current = work["current_a"].to_numpy(dtype=float)
    cell_v = work["voltage_v"].to_numpy(dtype=float) / cells_series
    temp = work["temperature_c"].to_numpy(dtype=float)

    crate = current / capacity_ah
    abs_crate = np.abs(crate)
    sign = np.sign(crate)
    slow = (abs_crate >= SWEEP_MIN_CRATE) & (abs_crate <= SWEEP_MAX_CRATE)

    dt_min = np.empty(len(work), dtype=float)
    dt_min[0] = 0.0
    dt_min[1:] = np.diff(ts).astype("timedelta64[s]").astype(float) / 60.0

    not_slow = ~slow
    prev_not_slow = np.concatenate([[True], not_slow[:-1]])
    flip = np.concatenate([[True], sign[1:] != sign[:-1]])
    gap = dt_min > SWEEP_GAP_MIN
    seg_start = not_slow | prev_not_slow | flip | gap

    # Contiguous segment boundaries — O(n) over the whole series.
    bounds = np.flatnonzero(seg_start)
    bounds = np.append(bounds, len(work))

    sweeps: list[Sweep] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b - a < SWEEP_MIN_ROWS:
            continue
        if not slow[a:b].all():        # segment opened on a non-slow row
            continue
        seg_dt_h = dt_min[a:b].copy()
        seg_dt_h[0] = 0.0
        seg_dt_h /= 60.0
        q = np.cumsum(np.abs(current[a:b]) * seg_dt_h)
        if q[-1] < SWEEP_MIN_DSOC * capacity_ah:
            continue
        sweeps.append(
            Sweep(
                ts0=pd.Timestamp(ts[a]),
                cell_v=cell_v[a:b],
                q_ah=q,
                temp_med=float(np.median(temp[a:b])),
            )
        )
    return sweeps


def anchored_capacity(
    cell_v: np.ndarray[Any, Any],
    q_ah: np.ndarray[Any, Any],
    v_lo: float,
    v_hi: float,
) -> float:
    """Charge [Ah] moved between two fixed cell-voltage anchors.

    Returns ``nan`` unless the sweep actually spans ``[v_lo, v_hi]`` — a
    partial sweep that never reaches an anchor can't measure it.
    """
    if cell_v.min() > v_lo or cell_v.max() < v_hi:
        return float("nan")
    order = np.argsort(cell_v)
    v_sorted = cell_v[order]
    q_sorted = q_ah[order]
    q_lo = float(np.interp(v_lo, v_sorted, q_sorted))
    q_hi = float(np.interp(v_hi, v_sorted, q_sorted))
    return abs(q_hi - q_lo)


def _ica_core(
    cell_v: np.ndarray[Any, Any], q_ah: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """Numpy ICA core: uniform voltage grid + |dq/dV|. ``None`` if unusable."""
    if cell_v.size < SWEEP_MIN_ROWS:
        return None
    order = np.argsort(cell_v)
    v_sorted = cell_v[order]
    q_sorted = q_ah[order]
    v_lo, v_hi = float(v_sorted[0]), float(v_sorted[-1])
    if v_hi - v_lo < 1e-3:
        return None
    window = SAVGOL_WINDOW if SAVGOL_WINDOW % 2 == 1 else SAVGOL_WINDOW - 1
    if window <= SAVGOL_POLYORDER:
        return None
    v_grid = np.linspace(v_lo, v_hi, GRID_POINTS)
    q_on_v = np.interp(v_grid, v_sorted, q_sorted)
    q_smooth = savgol_filter(q_on_v, window, SAVGOL_POLYORDER)
    ica = np.abs(np.gradient(q_smooth, v_grid))
    return v_grid, ica


def ica_dva_curve(cell_v: np.ndarray[Any, Any], q_ah: np.ndarray[Any, Any]) -> pd.DataFrame:
    """Public wrapper: one sweep's ICA curve as a frame (``v``, ``ica``).

    Empty if the sweep is too short or too flat to differentiate stably.
    """
    core = _ica_core(cell_v, q_ah)
    if core is None:
        return pd.DataFrame(columns=["v", "ica"])
    v_grid, ica = core
    return pd.DataFrame({"v": v_grid, "ica": ica})


def find_signature_peaks(
    v: np.ndarray[Any, Any], ica: np.ndarray[Any, Any]
) -> list[dict[str, float]]:
    """Locate the dominant ICA peaks of one sweep, strongest first."""
    if ica.size == 0 or not np.isfinite(ica).any():
        return []
    peak_max = float(np.nanmax(ica))
    if peak_max <= 0:
        return []
    idx, props = find_peaks(ica, prominence=PEAK_PROMINENCE_FRAC * peak_max)
    if idx.size == 0:
        return []
    proms = props["prominences"]
    order = np.argsort(proms)[::-1][:MAX_PEAKS]
    return [
        {"v": float(v[idx[i]]), "height": float(ica[idx[i]]), "prominence": float(proms[i])}
        for i in order
    ]


def _peak_features(sweep: Sweep) -> dict[str, float] | None:
    core = _ica_core(sweep.cell_v, sweep.q_ah)
    if core is None:
        return None
    peaks = find_signature_peaks(*core)
    if not peaks:
        return None
    return {
        "main_peak_v": peaks[0]["v"],
        "main_peak_height": peaks[0]["height"],
        "n_peaks": float(len(peaks)),
    }


def monthly_signatures(
    df: pd.DataFrame,
    capacity_ah: float,
    cells_series: int,
    chemistry: str = "LFP",
) -> pd.DataFrame:
    """Per-(month) ICA signature + anchored capacity for one system."""
    sweeps = reconstruct_sweeps(df, capacity_ah, cells_series)
    v_lo, v_hi = CHEM_ANCHORS.get(chemistry, _DEFAULT_ANCHORS)

    rows: list[dict[str, Any]] = []
    for sw in sweeps:
        rows.append({
            "month": sw.ts0.tz_localize(None).to_period("M").to_timestamp(),
            "anchored_cap_ah": anchored_capacity(sw.cell_v, sw.q_ah, v_lo, v_hi),
            "temp_med_c": sw.temp_med,
            "_sweep": sw,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "month", "n_sweeps", "anchored_cap_ah", "cap_cov",
            "main_peak_v", "main_peak_height", "n_peaks_med", "temp_med_c",
        ])

    per_sweep = pd.DataFrame(rows)
    out_rows: list[dict[str, Any]] = []
    for month, grp in per_sweep.groupby("month"):
        caps = grp["anchored_cap_ah"].dropna().to_numpy()
        # ICA features on a capped, evenly-spaced sample (the costly part).
        sample = grp["_sweep"].to_list()
        if len(sample) > MAX_ICA_PER_MONTH:
            pick = np.linspace(0, len(sample) - 1, MAX_ICA_PER_MONTH).astype(int)
            sample = [sample[i] for i in pick]
        feats = [f for f in (_peak_features(s) for s in sample) if f is not None]
        fdf = pd.DataFrame(feats) if feats else pd.DataFrame()
        out_rows.append({
            "month": month,
            "n_sweeps": int(len(grp)),
            "anchored_cap_ah": float(np.median(caps)) if caps.size else float("nan"),
            "cap_cov": float(np.std(caps) / np.mean(caps)) if caps.size > 2 else float("nan"),
            "main_peak_v": float(fdf["main_peak_v"].median()) if not fdf.empty else float("nan"),
            "main_peak_height": float(fdf["main_peak_height"].median()) if not fdf.empty else float("nan"),
            "n_peaks_med": float(fdf["n_peaks"].median()) if not fdf.empty else float("nan"),
            "temp_med_c": float(grp["temp_med_c"].median()),
        })
    return pd.DataFrame(out_rows).sort_values("month").reset_index(drop=True)


def _robust_fade(months: pd.Series, cap: pd.Series) -> tuple[float, float]:
    """Linear capacity-fade rate [%/yr] and R² from a smoothed trend."""
    valid = cap.notna()
    if valid.sum() < MIN_MONTHS:
        return float("nan"), float("nan")
    t = (months[valid] - months[valid].iloc[0]).dt.days.to_numpy() / 365.25
    y = cap[valid].to_numpy()
    base = float(np.median(y[:BASELINE_MONTHS]))
    if base <= 0:
        return float("nan"), float("nan")
    y_pct = y / base * 100.0
    slope, intercept = np.polyfit(t, y_pct, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((y_pct - pred) ** 2))
    ss_tot = float(np.sum((y_pct - y_pct.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-slope), r2   # fade is the negative slope (decline → +%/yr)


def attribute_modes(monthly: pd.DataFrame) -> pd.DataFrame:
    """Attribute month-over-month change to LLI vs LAM.

    Works off a 3-month rolling-median capacity trend to suppress the
    field noise, then splits: capacity loss with a stable peak voltage →
    LLI; peak-height loss and peak drift → LAM. A transparent heuristic,
    not a teardown.
    """
    out = monthly.copy()
    cols = ("cap_smooth_ah", "cap_fade_pct", "peak_v_shift_v",
            "peak_height_loss", "lli_frac", "lam_frac")
    if len(out) < MIN_MONTHS:
        for col in cols:
            out[col] = np.nan
        return out

    out["cap_smooth_ah"] = (
        out["anchored_cap_ah"].rolling(SMOOTH_MONTHS, min_periods=1, center=True).median()
    )
    base = out.head(BASELINE_MONTHS)
    base_cap = float(base["cap_smooth_ah"].median())
    base_peak_v = float(base["main_peak_v"].median())
    base_peak_h = float(base["main_peak_height"].median())

    out["cap_fade_pct"] = (1.0 - out["cap_smooth_ah"] / base_cap) * 100.0
    out["peak_v_shift_v"] = out["main_peak_v"] - base_peak_v
    out["peak_height_loss"] = (1.0 - out["main_peak_height"] / base_peak_h).clip(lower=0.0)

    fade = out["cap_fade_pct"].clip(lower=0.0) / 100.0
    drift = (out["peak_v_shift_v"].abs() / PEAK_V_SHIFT_REF).clip(0.0, 1.0)
    lli_score = fade * (1.0 - drift)
    lam_score = out["peak_height_loss"] + drift * fade
    total = lli_score + lam_score
    out["lli_frac"] = np.where(total > 1e-9, lli_score / total, np.nan)
    out["lam_frac"] = np.where(total > 1e-9, lam_score / total, np.nan)
    return out


def _system_summary(sid: str, chemistry: str, modes: pd.DataFrame) -> dict[str, object]:
    """One-line health verdict for a system from its monthly mode series."""
    fade, r2 = _robust_fade(modes["month"], modes.get("cap_smooth_ah", modes["anchored_cap_ah"]))
    valid = modes.dropna(subset=["lli_frac"])
    if valid.empty:
        dominant = "insufficient-data"
    else:
        lli, lam = float(valid["lli_frac"].mean()), float(valid["lam_frac"].mean())
        tag = "LLI" if lli >= lam else "LAM"
        dominant = f"{tag} ({max(lli, lam) * 100:.0f}%)"
    cap_cov = float(modes["cap_cov"].median()) if "cap_cov" in modes else float("nan")
    return {
        "system_id": sid,
        "chemistry": chemistry,
        "n_months": int(len(modes)),
        "peak_richness": round(float(modes["n_peaks_med"].median()), 1) if len(modes) else np.nan,
        "cap_cov": round(cap_cov, 3),
        "fade_pct_per_yr": round(fade, 2),
        "fade_r2": round(r2, 2),
        "dominant_mode": dominant,
    }


def main() -> None:
    if not IDENTITY_PATH.exists():
        raise SystemExit(
            f"missing {IDENTITY_PATH}. Run `python -m bess_fleet.pipeline.load_identity`."
        )
    ident = pd.read_parquet(IDENTITY_PATH)
    cap_lookup = dict(zip(ident["system_id"], ident["capacity_ah"], strict=True))
    cells_lookup = dict(zip(ident["system_id"], ident["cells_series"], strict=True))
    chem_lookup = dict(zip(ident["system_id"], ident["chemistry"], strict=True))

    files = sorted(BRONZE_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no bronze parquets in {BRONZE_DIR}.")

    print(f"degradation-mode estimation over {len(files)} systems\n", flush=True)
    all_modes: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for path in files:
        sid = path.stem
        if sid not in cap_lookup:
            print(f"  [{sid}] SKIP — no identity row", flush=True)
            continue
        df = pd.read_parquet(
            path, columns=["timestamp", "voltage_v", "current_a", "temperature_c"]
        )
        chemistry = str(chem_lookup.get(sid, "?"))
        monthly = monthly_signatures(
            df, float(cap_lookup[sid]), int(cells_lookup[sid]), chemistry
        )
        modes = attribute_modes(monthly)
        modes.insert(0, "chemistry", chemistry)
        modes.insert(0, "system_id", sid)
        all_modes.append(modes)
        summary = _system_summary(sid, chemistry, modes)
        summaries.append(summary)
        print(
            f"  [{sid}] {chemistry:<3} months={summary['n_months']:>3}  "
            f"peaks={summary['peak_richness']!s:>4}  cap-CoV={summary['cap_cov']!s:>5}  "
            f"fade={summary['fade_pct_per_yr']!s:>5}%/yr(R²={summary['fade_r2']})  "
            f"mode={summary['dominant_mode']}",
            flush=True,
        )

    if not all_modes:
        print("\nno systems processed")
        return
    out = pd.concat(all_modes, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_to_parquet(out, OUT_PATH, index=False, compression="snappy")
    print(f"\nwrote {OUT_PATH}: {len(out):,} (system × month) rows")

    print("\nPer-system summary — note how cap-CoV (capacity confidence)")
    print("and peak-richness split by chemistry:")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
