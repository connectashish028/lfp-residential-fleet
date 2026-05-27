"""Convert the six LFP raw zips into 1-minute-cadence parquet files.

Output: ``data/lfp_1min/ID{NN}.parquet`` — one file per system. Aggregates
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

    python -m bess_fleet.pipeline.lfp_to_1min_parquet

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

# Active scope: all six LFP systems in the Figgener fleet. Per-system
# KPIs (RTE, EFC, throughput, idle fraction, mean ΔT) stand on each
# rack's own record — no cross-rack alignment needed. Peer-comparison
# detection, which does need ≥3 hardware-identical racks, is restricted
# downstream to the 3-rack 8.09 kWh subgroup {ID14, ID16, ID17}; this
# script doesn't enforce that.
LFP_IDS: frozenset[str] = frozenset(
    {"ID14", "ID16", "ID17", "ID18", "ID19", "ID20"}
)
FREQ = "1min"
OUT_DIR = DATA_DIR / "lfp_1min"

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
    table = pa_csv.read_csv(
        _io.BytesIO(buf),
        convert_options=pa_csv.ConvertOptions(
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
    lfp_zips = [z for z in zips if _system_id_from_zip(z) in LFP_IDS]
    if not lfp_zips:
        raise SystemExit(f"no LFP zips matched {sorted(LFP_IDS)}")
    print(f"processing {len(lfp_zips)} LFP zips at {FREQ} cadence", flush=True)
    print(f"output: {OUT_DIR}\n", flush=True)

    total_rows = 0
    for zp in lfp_zips:
        try:
            df = _resample_zip(zp)
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"  [skip] {zp.name}: {exc}", flush=True)
            continue
        sid = df["system_id"].iloc[0]
        out_path = OUT_DIR / f"{sid}.parquet"
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
