"""Convert the raw per-system zips into 1-minute-cadence parquet files.

Output: ``data/bronze_1min/ID{NN}.parquet`` — one file per system. Aggregates
the 1-second raw measurements to 1-minute means. **No cleaning rules
applied** — the user wants the raw 1-min cadence as a starting point for
their own cleaning workflow.

Columns written per parquet:

* ``timestamp``           — UTC, 1-minute cadence
* ``system_id``           — e.g. ``"ID14"``; also encoded in the filename
* ``voltage_v``           — DC pack voltage (mean)
* ``current_a``           — DC pack current (mean, signed)
* ``power_kw``            — DC power = power_w / 1000 (mean, signed)
* ``temperature_c``       — battery surface temperature (mean)
* ``ambient_c``           — room temperature (mean — raw, may contain
                            Figgener's -100 °C sentinel which the user
                            will clean downstream)
* ``interpolated_frac``   — fraction of source 1-second samples in this
                            1-minute window that were flagged
                            ``Interpolated == 1`` in the raw CSVs. Range
                            [0.0, 1.0]. The Figgener provenance flag —
                            keep it so the cleaning workflow can drop or
                            de-weight windows that are largely
                            reconstructed rather than measured.

This script does *not* import the production CSV reader (which omits
``Interpolated``); it has its own local reader so the production build
pipeline stays untouched.

Run with:

    python -m bess_fleet.pipeline.raw_to_1min_parquet

Expected run time: ~30-40 min on a laptop (CSV read + date parse dominate).
"""

from __future__ import annotations

import io as _io
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

from bess_fleet.db import DATA_DIR
from bess_fleet.io import safe_to_parquet

# ─── helpers inlined (were previously imported from bess_fleet.build) ──────
_MONTH_PATTERN = re.compile(r"(\d{4})_(\d{2})_System_ID_\d+\.csv$")
_ZIP_PATTERN = re.compile(r"Data_ID_(\d+)\.zip$")


def _system_id_from_zip(path: Path) -> str:
    m = _ZIP_PATTERN.search(path.name)
    if not m:
        raise ValueError(f"unrecognised zip name: {path}")
    return f"ID{int(m.group(1)):02d}"


def _months(zip_members: Iterable[str]) -> list[str]:
    """Return monthly-CSV member names sorted chronologically."""
    keep: list[tuple[pd.Timestamp, str]] = []
    for name in zip_members:
        m = _MONTH_PATTERN.search(name)
        if not m:
            continue
        ts = pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        keep.append((ts, name))
    return [n for _, n in sorted(keep)]
# ──────────────────────────────────────────────────────────────────────────

# Active scope, grouped by chemistry. The Figgener dataset (Zenodo
# 12091223) spans 21 systems across three cathode chemistries; this is
# the subset we ingest into the lake. The cross-chemistry mix is
# deliberate: the diagnostics downstream are validated to behave
# differently on a flat-OCV LFP plateau vs a sloped NMC/LMO curve, and
# that contrast is the point.
#
# Per-system KPIs (RTE, EFC, throughput, idle fraction, mean ΔT) stand
# on each rack's own record — no cross-rack alignment needed.
LFP_E:     frozenset[str] = frozenset({"ID14", "ID16", "ID17", "ID18", "ID19", "ID20"})  # Mfr E · LFP
LMO_NMC_A: frozenset[str] = frozenset({"ID01", "ID02"})                                  # Mfr A · LMO/NMC blend
NMC_BC:    frozenset[str] = frozenset({"ID07", "ID11"})                                  # Mfr B/C · pure NMC
SYSTEM_IDS: frozenset[str] = LFP_E | LMO_NMC_A | NMC_BC
FREQ = "1min"
OUT_DIR = DATA_DIR / "bronze_1min"

# Local rename map — extends the production one with Interpolated.
_CSV_COLS_RENAME = {
    "Time": "timestamp",
    "P_in_W": "power_w",
    "V_in_V": "voltage_v",
    "I_in_A": "current_a",
    "T_Bat_in_C": "temperature_c",
    "T_Room_in_C": "ambient_c",
    "Interpolated": "interpolated_flag",
}

_PA_TYPES = pa.schema(
    [
        ("Time",         pa.string()),
        ("P_in_W",       pa.float32()),
        ("V_in_V",       pa.float32()),
        ("I_in_A",       pa.float32()),
        ("T_Bat_in_C",   pa.float32()),
        ("T_Room_in_C",  pa.float32()),
        ("Interpolated", pa.int8()),
    ]
).empty_table().schema


def _read_csv_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Read one monthly CSV including the Interpolated provenance flag."""
    with zf.open(name) as fh:
        buf = fh.read()
    table = pa_csv.read_csv(  # type: ignore[attr-defined]
        _io.BytesIO(buf),
        convert_options=pa_csv.ConvertOptions(  # type: ignore[attr-defined]
            column_types={
                c: t for c, t in zip(_PA_TYPES.names, _PA_TYPES.types, strict=True)
            },
            include_columns=list(_CSV_COLS_RENAME),
        ),
    )
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    df["Time"] = pd.to_datetime(df["Time"], format="%d-%b-%Y %H:%M:%S")
    return df.rename(columns=_CSV_COLS_RENAME)


def _resample_zip(zip_path: Path) -> pd.DataFrame:
    """Stream one system zip → 1-minute mean DataFrame (no cleaning).

    The Interpolated flag is averaged across each 1-min window, producing
    ``interpolated_frac`` ∈ [0, 1]. All other channels are simple means.
    """
    sid = _system_id_from_zip(zip_path)
    pieces: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = _months(zf.namelist())
        for name in members:
            month_df = _read_csv_member(zf, name)
            month_df = (
                month_df.set_index("timestamp")
                .sort_index()
                .resample(FREQ)
                .mean(numeric_only=True)
                .dropna(how="all")
                .reset_index()
                .rename(columns={"interpolated_flag": "interpolated_frac"})
            )
            pieces.append(month_df)
        print(f"  [{sid}] {len(members)} months read", flush=True)
    df = pd.concat(pieces, ignore_index=True).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["system_id"] = sid
    df["power_kw"] = (df["power_w"] / 1000.0).astype("float32")
    return df.drop(columns=["power_w"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zips = sorted((DATA_DIR / "raw").glob("Data_ID_*.zip"))
    in_scope = [z for z in zips if _system_id_from_zip(z) in SYSTEM_IDS]
    if not in_scope:
        raise SystemExit(f"no zips matched {sorted(SYSTEM_IDS)}")
    print(f"{len(in_scope)} systems in scope at {FREQ} cadence", flush=True)
    print(f"output: {OUT_DIR}\n", flush=True)

    total_rows = 0
    for zp in in_scope:
        sid = _system_id_from_zip(zp)
        out_path = OUT_DIR / f"{sid}.parquet"
        # Incremental: a system's bronze parquet is a deterministic
        # function of its raw zip, so skip systems already resampled
        # unless the zip is newer. raw→1-min resampling is by far the
        # costliest step (1-second data), so this makes adding a system
        # cheap instead of reprocessing the whole fleet.
        if out_path.exists() and out_path.stat().st_mtime >= zp.stat().st_mtime:
            print(f"  [{sid}] skip — {out_path.name} already up to date", flush=True)
            continue
        try:
            df = _resample_zip(zp)
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"  [skip] {zp.name}: {exc}", flush=True)
            continue
        safe_to_parquet(df, out_path, index=False, compression="snappy")
        size_mb = out_path.stat().st_size / 1e6
        interp_mean = float(df["interpolated_frac"].mean())
        print(
            f"  [{sid}] wrote {out_path.name}: {len(df):,} rows, "
            f"{size_mb:.1f} MB, mean interpolated_frac = {interp_mean:.3f}",
            flush=True,
        )
        total_rows += len(df)

    print(
        f"\ndone. {total_rows:,} rows total across "
        f"{len(list(OUT_DIR.glob('*.parquet')))} files in {OUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
