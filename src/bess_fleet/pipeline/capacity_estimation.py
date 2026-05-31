"""Usable-capacity / SOHc estimation from field data — a faithful
replication of Figgener et al. (*Nature Energy* 2024,
``10.1038/s41560-024-01620-9``), with an **uncertainty-quantification**
twist.

This is the paper's *core* method (absolute usable capacity → SOHc →
ageing rate), distinct from :mod:`.degradation_modes` (the ICA/DVA
mechanism follow-up). It is the rigorous replacement for the crude SoH
that was removed earlier.

Method (paper eq. 1–3)
----------------------
1. **Rest detection** — find low-current relaxation phases and classify
   each as *top* (post-charge, near-full / EOC) or *bottom* (post-
   discharge, near-empty / EOD) using per-system adaptive voltage
   percentiles. EOC/EOD drift over life (the BMS lowers EOD to mask
   ageing), so we read them from behaviour rather than hardcoding.
2. **OCV via a 2nd-order ECM** (eq. 1) — for each rest, fit
   ``V(t) = V_OCV + V_fast·e^(−t/τ_fast) + V_slow·e^(−t/τ_slow)`` to the
   *1-second* relaxation (the fast charge-transfer term is invisible at
   1-min cadence) with a two-step bounded fit, extracting the V_OCV
   asymptote **and its standard error** from the fit covariance.
3. **OCV → SOC** on the chemistry curve, carrying the local slope
   ``dSOC/dOCV`` — this is where the twist bites: a flat LFP plateau has
   a huge slope, so the same σ_OCV becomes a huge σ_SOC. The error model
   therefore *reproduces the paper's accuracy ranking* (LMO/NMC sharp →
   tight; LFP flat → wide) as a consequence, not an assumption.
4. **Offset-current correction** (eq. 2) — the DC-pole meter misses the
   BMS self-supply + balancing draw, so a closed cycle doesn't integrate
   to zero. Solve ``I_offset`` by forcing same-state (top→top / bottom→
   bottom) cycles to net ≈0 charge.
5. **Capacity** — integrate offset-corrected current between a top and a
   bottom rest, normalised by the SOC swing:
   ``C_usable = ΔQ_corrected / |SOC_top − SOC_bottom|``.
6. **SOHc = C_usable / C_nominal** (eq. 3), with σ propagated from
   σ_SOC(top, bottom) and σ_offset.
7. **Ageing rate** — inverse-variance-weighted linear fit of SOHc vs
   time; the gradient is the %/yr fade, reported with a fit CI that
   inherits the per-estimate uncertainties.

Outputs
-------
* ``data/curated/capacity_estimates.parquet`` — one row per estimate
  (system, timestamp, soh_pct, sigma_soh_pp, integration mode).
* ``data/curated/capacity_soh.parquet`` — one row per system (ageing
  rate ± CI, mean σ, n estimates, BOL/EOL projections).

Run with::

    python -m bess_fleet.pipeline.capacity_estimation
"""

from __future__ import annotations

import io as _io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
from scipy.optimize import curve_fit

from bess_fleet.db import DATA_DIR
from bess_fleet.io import safe_to_parquet
from bess_fleet.pipeline.derive_soc import _DEFAULT_CHEMISTRY, OCV_SOC_TABLES
from bess_fleet.pipeline.raw_to_1min_parquet import _MONTH_PATTERN

PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
IDENTITY_PATH = DATA_DIR / "identity.parquet"
EST_OUT_PATH = DATA_DIR / "curated" / "capacity_estimates.parquet"
SOH_OUT_PATH = DATA_DIR / "curated" / "capacity_soh.parquet"

# ─── Rest / relaxation detection ───────────────────────────────────────────
REST_POWER_W   = 30.0     # |P| below this ≈ resting (BMS/standby only)
REST_MIN_MIN   = 20       # a usable relaxation lasts at least this long
REST_GAP_MIN   = 3.0      # >this between 1-min samples ends a rest
TOP_PCTL       = 80.0     # rest cell-V ≥ this system percentile → near-full
BOTTOM_PCTL    = 20.0     # rest cell-V ≤ this system percentile → near-empty
MIN_REST_CELL_V = 2.0     # below this the cell is 'off'/disconnected, not at rest

# ─── 2nd-order ECM bounds (eq. 1) ──────────────────────────────────────────
TAU_FAST_BOUNDS = (3.0, 180.0)      # s — charge transfer
TAU_SLOW_BOUNDS = (120.0, 7200.0)   # s — diffusion
FAST_SPLIT_S    = 90.0              # tail after this ≈ fast-RC-free (step 1)
ECM_MIN_POINTS  = 30               # need this many 1-s samples to fit
ECM_MIN_POINTS_1MIN = 15           # ~15 min of 1-min samples is enough for the
                                   # asymptote (the fast RC is already gone)

# ─── Capacity gating ───────────────────────────────────────────────────────
MIN_SOC_SWING  = 0.40     # only trust estimates spanning ≥40 % SOC
MIN_MONTHS_FIT = 4        # below this, no ageing rate is published

# Reliability gate for the published ageing rate (per system).
MIN_RELIABLE_EST   = 20      # need a decent number of full-cycle estimates
SIGMA_MAX_PP       = 8.0     # mean per-estimate σ above this → too noisy
CI95_MAX_PP_YR     = 1.5     # ageing-rate 95 % CI above this → untrustworthy
FADE_PLAUSIBLE_PP_YR = (-0.5, 10.0)   # outside this is physically implausible


@dataclass
class Rest:
    """One detected relaxation phase."""

    start: pd.Timestamp
    end: pd.Timestamp
    kind: str            # "top" | "bottom"
    cum_ah_end: float    # running coulomb count at the rest's end
    cell_v_1min: float   # mean cell voltage over the rest (1-min, pre-OCV-fit)
    ocv: float = float("nan")        # ECM-fitted rested OCV (set after 1-s fit)
    sigma_ocv: float = float("nan")  # standard error of the OCV asymptote


# ─── OCV → SOC with local slope (the uncertainty hinge) ─────────────────────
def ocv_to_soc_with_slope(ocv_v: float, chemistry: str) -> tuple[float, float]:
    """Map a cell OCV to SOC [fraction] and return the local
    ``dSOC/dOCV`` [1/V]. The slope is what turns σ_OCV into σ_SOC — large
    on a flat LFP plateau, small on a sloped NMC curve.
    """
    table = OCV_SOC_TABLES.get(chemistry, OCV_SOC_TABLES[_DEFAULT_CHEMISTRY])
    v, soc_tbl = table[:, 1], table[:, 0]
    eps = 1e-3
    soc_lo = float(np.interp(ocv_v - eps, v, soc_tbl))
    soc_hi = float(np.interp(ocv_v + eps, v, soc_tbl))
    soc = float(np.interp(ocv_v, v, soc_tbl)) / 100.0
    dsoc_docv = (soc_hi - soc_lo) / (2 * eps) / 100.0   # fraction per volt
    return soc, abs(dsoc_docv)


# ─── 2nd-order ECM relaxation fit (eq. 1) ──────────────────────────────────
def _ecm_2nd_order(
    t: np.ndarray[Any, Any], v_ocv: float, a_f: float, tau_f: float,
    a_s: float, tau_s: float,
) -> np.ndarray[Any, Any]:
    return v_ocv + a_f * np.exp(-t / tau_f) + a_s * np.exp(-t / tau_s)


def fit_relaxation_ocv(
    t_s: np.ndarray[Any, Any], v_cell: np.ndarray[Any, Any],
    min_points: int = ECM_MIN_POINTS,
) -> tuple[float, float, float] | None:
    """Two-step bounded fit of the 2nd-order ECM to one relaxation.

    Parameters
    ----------
    t_s
        Seconds since the rest started (1-second cadence, increasing).
    v_cell
        Per-cell terminal voltage over the rest.

    Returns
    -------
    ``(V_OCV, sigma_V_OCV, rmse)`` or ``None`` if the rest is too short /
    flat / the fit fails. ``sigma_V_OCV`` is the standard error of the
    asymptote from the fit covariance — the seed of the whole uncertainty
    chain.
    """
    if t_s.size < min_points or t_s[-1] - t_s[0] < REST_MIN_MIN * 60 * 0.5:
        return None
    t = t_s - t_s[0]
    v = v_cell.astype(float)
    v_last = float(np.median(v[-max(3, v.size // 10):]))   # asymptote guess

    # Step 1 — slow tail (fast RC decayed): single exponential → V_OCV0.
    tail = t >= FAST_SPLIT_S
    if tail.sum() >= 10:
        try:
            p0 = [v_last, v[tail][0] - v_last, 600.0]
            popt_s, _ = curve_fit(
                lambda tt, voc, a, tau: voc + a * np.exp(-tt / tau),
                t[tail] - t[tail][0], v[tail], p0=p0,
                bounds=([v.min() - 0.2, -1.0, TAU_SLOW_BOUNDS[0]],
                        [v.max() + 0.2, 1.0, TAU_SLOW_BOUNDS[1]]),
                maxfev=5000,
            )
            v_ocv0, a_s0, tau_s0 = popt_s
        except (RuntimeError, ValueError):
            v_ocv0, a_s0, tau_s0 = v_last, v[-1] - v_last, 600.0
    else:
        v_ocv0, a_s0, tau_s0 = v_last, v[-1] - v_last, 600.0

    # Step 2 — full 2nd-order with step-1 seeds.
    a_f0 = float(v[0] - v_ocv0 - a_s0)
    try:
        popt, pcov = curve_fit(
            _ecm_2nd_order, t, v,
            p0=[v_ocv0, a_f0, 30.0, a_s0, tau_s0],
            bounds=(
                [v.min() - 0.2, -2.0, TAU_FAST_BOUNDS[0], -2.0, TAU_SLOW_BOUNDS[0]],
                [v.max() + 0.2,  2.0, TAU_FAST_BOUNDS[1],  2.0, TAU_SLOW_BOUNDS[1]],
            ),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None

    v_ocv = float(popt[0])
    sigma = float(np.sqrt(abs(pcov[0, 0]))) if np.all(np.isfinite(pcov)) else float("nan")
    resid = v - _ecm_2nd_order(t, *popt)
    rmse = float(np.sqrt(np.mean(resid**2)))
    # A failed/unconstrained fit shows up as a non-finite or absurd sigma.
    if not np.isfinite(sigma) or sigma > 0.5:
        sigma = max(rmse, 5e-3)
    return v_ocv, sigma, rmse


# ─── Rest detection on the 1-min frame ─────────────────────────────────────
def detect_rest_windows(
    df: pd.DataFrame, capacity_ah: float, cells_series: int
) -> list[Rest]:
    """Find top/bottom relaxation windows + the running coulomb count.

    Operates on the cleaned 1-min frame; the ECM fit later pulls 1-second
    data for each returned window.
    """
    if df.empty:
        return []
    work = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(work["timestamp"]).dt.tz_localize(None)
    # NaN-fill the current before integrating — a single NaN poisons the
    # whole cumulative sum downstream (this is what zeroed ID02's offset).
    current = np.nan_to_num(work["current_a"].to_numpy(dtype=float), nan=0.0)
    cell_v = work["voltage_v"].to_numpy(dtype=float) / cells_series
    power_w = work["power_kw"].to_numpy(dtype=float) * 1000.0

    # Running coulomb count [Ah], gap-aware.
    dt_h = np.empty(len(work))
    dt_h[0] = 0.0
    dt_h[1:] = np.diff(ts.to_numpy()).astype("timedelta64[s]").astype(float) / 3600.0
    dt_h = np.clip(dt_h, 0.0, REST_GAP_MIN / 60.0)   # don't integrate across gaps
    cum_ah = np.cumsum(current * dt_h)

    # A real relaxation needs low power AND a plausible cell voltage —
    # exclude 'system off' / disconnected samples sitting near 0 V (these
    # are resting by power but are not battery relaxations).
    resting = (
        (np.abs(power_w) < REST_POWER_W)
        & np.isfinite(cell_v)
        & (cell_v > MIN_REST_CELL_V)
    )
    # Segment contiguous rest runs (break on non-rest or a data gap).
    gap = np.concatenate([[True], (np.diff(ts.to_numpy()).astype("timedelta64[s]")
                                   .astype(float) / 60.0) > REST_GAP_MIN])
    seg_start = (~resting) | np.concatenate([[True], ~resting[:-1]]) | gap

    # System-adaptive top/bottom thresholds. nan-percentile so a stray NaN
    # cell-voltage can't blank the thresholds (this zeroed ID18's rests).
    rest_v = cell_v[resting]
    if rest_v.size < 50:
        return []
    v_top = float(np.nanpercentile(rest_v, TOP_PCTL))
    v_bot = float(np.nanpercentile(rest_v, BOTTOM_PCTL))

    # Linear-time: walk contiguous segments via their boundaries; a
    # segment is a rest iff every row in it is resting. (No per-segment
    # full-array rescans.)
    ts_arr = ts.to_numpy()
    bounds = np.flatnonzero(seg_start)
    bounds = np.append(bounds, len(work))
    rests: list[Rest] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if (b - a) < REST_MIN_MIN or not resting[a:b].all():
            continue
        dur_min = (ts_arr[b - 1] - ts_arr[a]).astype("timedelta64[s]").astype(float) / 60.0
        if dur_min < REST_MIN_MIN:
            continue
        v_mean = float(np.mean(cell_v[a:b]))
        if v_mean >= v_top:
            kind = "top"
        elif v_mean <= v_bot:
            kind = "bottom"
        else:
            continue
        rests.append(Rest(
            start=pd.Timestamp(ts_arr[a]), end=pd.Timestamp(ts_arr[b - 1]), kind=kind,
            cum_ah_end=float(cum_ah[b - 1]), cell_v_1min=v_mean,
        ))
    return rests


# ─── 1-second window loader (hybrid: only the detected rests) ──────────────
def _zip_for_system(sid: str) -> Path:
    """Raw zip path for a system id (``ID07`` → ``Data_ID_07.zip``)."""
    return RAW_DIR / f"Data_ID_{sid[2:]}.zip"


def _load_month_1s(sid: str, year: int, month: int) -> pd.DataFrame | None:
    """Read one month's 1-second V/I for a system from its raw zip.

    Lean: only the three columns the OCV fit needs (Time, V, I), so the
    pyarrow read + timestamp parse stay cheap despite ~2.6 M rows/month.
    """
    zpath = _zip_for_system(sid)
    if not zpath.exists():
        return None
    member = f"{sid[2:]}/{year}_{month:02d}_System_ID_{sid[2:]}.csv"
    try:
        with zipfile.ZipFile(zpath) as zf:
            if member not in zf.namelist():
                hits = [n for n in zf.namelist()
                        if _MONTH_PATTERN.search(n) and f"{year}_{month:02d}" in n]
                if not hits:
                    return None
                member = hits[0]
            with zf.open(member) as fh:
                buf = fh.read()
        # Lean read — only the 3 columns the OCV fit needs.
        table = pa_csv.read_csv(  # type: ignore[attr-defined]
            _io.BytesIO(buf),
            convert_options=pa_csv.ConvertOptions(  # type: ignore[attr-defined]
                include_columns=["Time", "V_in_V", "I_in_A"],
            ),
        )
    except (zipfile.BadZipFile, OSError, KeyError, ValueError, pa.ArrowInvalid):
        return None
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["Time"], format="%d-%b-%Y %H:%M:%S")
    return df.rename(columns={"V_in_V": "voltage_v", "I_in_A": "current_a"})[
        ["timestamp", "voltage_v", "current_a"]
    ]


def _even_sample(items: list[Rest], n: int) -> list[Rest]:
    """Keep ``n`` evenly-spaced items (preserve time spread)."""
    if n <= 0 or len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    return [items[i] for i in idx]


def fit_rests_ocv(
    rests: list[Rest], sid: str, cells_series: int,
    month_stride: int = 1, per_month_cap: int = 8,
) -> None:
    """Fit the 2nd-order ECM OCV for each rest in place.

    Pulls 1-second data per month so each raw monthly CSV is read at most
    once. ``month_stride`` subsamples months (the 1-second read is the
    dominant cost) and ``per_month_cap`` bounds fits per month — both keep
    a time spread; a stride > 1 trades a denser SOHc series for speed.
    """
    by_month: dict[tuple[int, int], list[Rest]] = {}
    for r in rests:
        key = (r.start.year, r.start.month)
        by_month.setdefault(key, []).append(r)

    kept = sorted(by_month)[::max(1, month_stride)]
    for (year, month) in kept:
        month_rests = by_month[(year, month)]
        if len(month_rests) > per_month_cap:
            half = per_month_cap // 2
            tops = _even_sample([r for r in month_rests if r.kind == "top"], half)
            bots = _even_sample(
                [r for r in month_rests if r.kind == "bottom"], per_month_cap - half
            )
            month_rests = tops + bots
        df1s = _load_month_1s(sid, year, month)
        if df1s is None or df1s.empty:
            continue
        ts = df1s["timestamp"].to_numpy()
        for r in month_rests:
            mask = (ts >= np.datetime64(r.start)) & (ts <= np.datetime64(r.end))
            if mask.sum() < ECM_MIN_POINTS:
                continue
            sub = df1s.loc[mask]
            t_s = (sub["timestamp"] - r.start).dt.total_seconds().to_numpy()
            v_cell = sub["voltage_v"].to_numpy(dtype=float) / cells_series
            fit = fit_relaxation_ocv(t_s, v_cell)
            if fit is not None:
                r.ocv, r.sigma_ocv, _ = fit


def fit_rests_ocv_1min(rests: list[Rest], df_1min: pd.DataFrame, cells_series: int) -> None:
    """Fit each rest's OCV from the **1-minute** slice — no raw I/O.

    Validated to match the 1-second 2nd-order fit to ~0.1 mV (mean) on
    real rests: the fast charge-transfer RC has fully decayed before the
    first 1-min sample, so the V_OCV *asymptote* — all the capacity
    estimate needs — is identical. This is what makes a fleet-scale run
    tractable; :func:`fit_rests_ocv` (the 1-second path) remains for
    spot-validation. Uses ``searchsorted`` so it stays O(rests·log n).
    """
    ts = (
        pd.to_datetime(df_1min["timestamp"], utc=True)
        .dt.tz_localize(None).to_numpy()
    )
    v_cell = df_1min["voltage_v"].to_numpy(dtype=float) / cells_series
    for r in rests:
        i = int(np.searchsorted(ts, np.datetime64(r.start), "left"))
        j = int(np.searchsorted(ts, np.datetime64(r.end), "right"))
        if j - i < ECM_MIN_POINTS_1MIN:
            continue
        t_s = (ts[i:j] - ts[i]).astype("timedelta64[s]").astype(float)
        fit = fit_relaxation_ocv(t_s, v_cell[i:j], min_points=ECM_MIN_POINTS_1MIN)
        if fit is not None:
            r.ocv, r.sigma_ocv, _ = fit


# ─── Offset-current correction (eq. 2) ─────────────────────────────────────
def estimate_offset_current(rests: list[Rest]) -> tuple[float, float]:
    """Solve I_offset by forcing same-state cycles to net ≈0 charge.

    For a top→top (or bottom→bottom) cycle the battery returns to the same
    SOC, so ∫I_bat dt ≈ I_offset·Δt. Robust median over all such cycles;
    σ from the MAD.
    """
    offs: list[float] = []
    for a, b in zip(rests[:-1], rests[1:], strict=True):
        if a.kind != b.kind:
            continue
        dt_h = (b.end - a.end).total_seconds() / 3600.0
        if dt_h <= 1.0:
            continue
        offs.append((b.cum_ah_end - a.cum_ah_end) / dt_h)
    if not offs:
        return 0.0, 0.0
    arr = np.asarray(offs)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, 1.4826 * mad   # MAD→σ for a normal


# ─── Capacity estimate + uncertainty propagation ───────────────────────────
def estimate_capacity(
    rests: list[Rest], chemistry: str, capacity_ah: float,
    i_offset: float, sigma_offset: float,
) -> list[dict[str, Any]]:
    """One C_usable estimate per adjacent top↔bottom rest pair, with σ
    propagated from σ_OCV (via the OCV slope) and σ_offset.

    Adjacency is taken over the *OCV-fitted* rests, so month-sampling
    (which fits only a subset) doesn't sever the top→bottom chain.
    """
    fitted = [r for r in rests if np.isfinite(r.ocv)]
    out: list[dict[str, Any]] = []
    for a, b in zip(fitted[:-1], fitted[1:], strict=True):
        if a.kind == b.kind:
            continue
        top, bottom = (a, b) if a.kind == "top" else (b, a)
        soc_top, slope_top = ocv_to_soc_with_slope(top.ocv, chemistry)
        soc_bot, slope_bot = ocv_to_soc_with_slope(bottom.ocv, chemistry)
        swing = abs(soc_top - soc_bot)
        if swing < MIN_SOC_SWING:
            continue

        dt_h = (b.end - a.end).total_seconds() / 3600.0
        dq_corr = (b.cum_ah_end - a.cum_ah_end) - i_offset * dt_h
        c_usable = abs(dq_corr) / swing
        if not (0.5 * capacity_ah <= c_usable <= 1.25 * capacity_ah):
            continue   # physically implausible → reject

        sig_soc_top = top.sigma_ocv * slope_top
        sig_soc_bot = bottom.sigma_ocv * slope_bot
        sig_swing = float(np.hypot(sig_soc_top, sig_soc_bot))
        sig_c = float(np.hypot(
            c_usable / swing * sig_swing,
            sigma_offset * abs(dt_h) / swing,
        ))
        out.append({
            "timestamp": b.end,
            "mode": "F2E" if a.kind == "top" else "E2F",
            "capacity_ah": c_usable,
            "soc_swing": swing,
            "soh_pct": c_usable / capacity_ah * 100.0,
            "sigma_soh_pp": sig_c / capacity_ah * 100.0,
        })
    return out


# ─── Inverse-variance-weighted ageing fit ──────────────────────────────────
def weighted_ageing_fit(est: pd.DataFrame, sid: str, chemistry: str) -> dict[str, Any]:
    """Weighted linear fit of SOHc vs time → ageing rate ± CI."""
    base = {
        "system_id": sid, "chemistry": chemistry, "n_estimates": int(len(est)),
        "ageing_pct_per_yr": float("nan"), "ageing_ci95_pp_yr": float("nan"),
        "mean_sigma_pp": float("nan"), "ci75_width_pp": float("nan"),
        "soh_latest_pct": float("nan"), "reliable": False,
    }
    if est.empty:
        return base
    e = est.sort_values("timestamp").reset_index(drop=True)
    t0 = e["timestamp"].iloc[0]
    t_yr = (e["timestamp"] - t0).dt.total_seconds().to_numpy() / (365.25 * 86400)
    soh = e["soh_pct"].to_numpy(dtype=float)
    sig = e["sigma_soh_pp"].to_numpy(dtype=float)
    sig = np.where(np.isfinite(sig) & (sig > 0.1), sig, np.nanmedian(sig[sig > 0]) if (sig > 0).any() else 1.0)

    n_months = int(e["timestamp"].dt.to_period("M").nunique())
    mean_sigma = round(float(np.mean(sig)), 2)
    base["mean_sigma_pp"] = mean_sigma
    base["soh_latest_pct"] = round(float(np.median(soh[-3:])), 1)
    # 75 % band of estimates around their median (paper-style CI width).
    base["ci75_width_pp"] = round(float(np.percentile(soh, 87.5) - np.percentile(soh, 12.5)), 1)
    if n_months < MIN_MONTHS_FIT or len(e) < 4:
        return base

    coeffs, cov = np.polyfit(t_yr, soh, 1, w=1.0 / sig, cov=True)
    slope, _intercept = coeffs
    se_slope = float(np.sqrt(abs(cov[0, 0])))
    fade = float(-slope)
    ci95 = float(1.96 * se_slope)
    base["ageing_pct_per_yr"] = round(fade, 2)
    base["ageing_ci95_pp_yr"] = round(ci95, 2)
    # Reliability gate — an estimate set is only trustworthy when there are
    # enough clean full cycles. Like the degradation module, the framework
    # flags where it can't be believed rather than emitting a confident
    # wrong number. (The per-estimate σ does the heavy lifting here: it
    # flags noisy systems — e.g. ID01 — even within a "good" chemistry.)
    base["reliable"] = bool(
        len(e) >= MIN_RELIABLE_EST
        and FADE_PLAUSIBLE_PP_YR[0] <= fade <= FADE_PLAUSIBLE_PP_YR[1]
        and mean_sigma <= SIGMA_MAX_PP
        and ci95 <= CI95_MAX_PP_YR
    )
    return base


def main() -> None:
    if not IDENTITY_PATH.exists():
        raise SystemExit(f"missing {IDENTITY_PATH}. Run load_identity first.")
    ident = pd.read_parquet(IDENTITY_PATH)
    cap = dict(zip(ident["system_id"], ident["capacity_ah"], strict=True))
    cells = dict(zip(ident["system_id"], ident["cells_series"], strict=True))
    chem = dict(zip(ident["system_id"], ident["chemistry"], strict=True))

    files = sorted(PROCESSED_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no cleaned parquets in {PROCESSED_DIR}.")

    print(f"capacity / SOHc estimation over {len(files)} systems\n", flush=True)
    all_est: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for path in files:
        sid = path.stem
        if sid not in cap:
            continue
        df = pd.read_parquet(
            path, columns=["timestamp", "voltage_v", "current_a", "power_kw"]
        )
        rests = detect_rest_windows(df, float(cap[sid]), int(cells[sid]))
        fit_rests_ocv_1min(rests, df, int(cells[sid]))
        i_off, sig_off = estimate_offset_current(rests)
        ests = estimate_capacity(rests, str(chem[sid]), float(cap[sid]), i_off, sig_off)
        est_df = pd.DataFrame(ests)
        if not est_df.empty:
            est_df.insert(0, "chemistry", str(chem[sid]))
            est_df.insert(0, "system_id", sid)
            all_est.append(est_df)
        summary = weighted_ageing_fit(est_df, sid, str(chem[sid]))
        summaries.append(summary)
        print(
            f"  [{sid}] {chem[sid]:<3} rests={len(rests):>4} "
            f"estimates={summary['n_estimates']:>4}  "
            f"fade={summary['ageing_pct_per_yr']!s:>5}±{summary['ageing_ci95_pp_yr']!s} pp/yr  "
            f"σ̄={summary['mean_sigma_pp']!s} pp  CI75={summary['ci75_width_pp']!s} pp",
            flush=True,
        )

    EST_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if all_est:
        safe_to_parquet(pd.concat(all_est, ignore_index=True), EST_OUT_PATH,
                        index=False, compression="snappy")
        print(f"\nwrote {EST_OUT_PATH}")
    summary_df = pd.DataFrame(summaries)
    safe_to_parquet(summary_df, SOH_OUT_PATH, index=False, compression="snappy")
    print(f"wrote {SOH_OUT_PATH}: {len(summary_df)} systems\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

