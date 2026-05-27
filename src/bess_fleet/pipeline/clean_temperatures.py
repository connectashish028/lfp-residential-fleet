"""Clean temperature sentinel values from the 1-min LFP parquets.

Reads raw per-system parquets from ``data/lfp_1min/`` and writes the
cleaned versions to ``data/processed/``. The raw layer is **untouched**
so the original Figgener sentinels are preserved for audit.

Cleaning rules:

* ``ambient_c < -30 °C``  → ``NaN``  (Figgener uses -100 as sensor-offline sentinel)
* ``ambient_c == 0.0``    → ``NaN``  (defensive — exact 0.0 indoor is suspicious)
* ``temperature_c < -30 °C`` → ``NaN``  (same sentinel on the battery-surface sensor)

No upper-bound clipping: values above 60–80 °C might be real thermal
events worth flagging downstream, not faked away here.

Run with::

    python -m bess_fleet.pipeline.clean_temperatures
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_fleet.db import DATA_DIR
from bess_fleet.io import safe_to_parquet

RAW_DIR = DATA_DIR / "lfp_1min"
OUT_DIR = DATA_DIR / "processed"


def clean_temperatures(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Apply the sentinel cleaning rule. Return (cleaned_df, n_amb_fixed, n_tbat_fixed)."""
    out = df.copy()
    bad_amb  = (out["ambient_c"]     < -30) | (out["ambient_c"] == 0.0)
    bad_tbat = (out["temperature_c"] < -30)
    out.loc[bad_amb,  "ambient_c"]     = np.nan
    out.loc[bad_tbat, "temperature_c"] = np.nan
    return out, int(bad_amb.sum()), int(bad_tbat.sum())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        raise SystemExit(f"no raw parquets in {RAW_DIR}")
    print(f"cleaning {len(raw_files)} files: {RAW_DIR}  →  {OUT_DIR}\n", flush=True)

    total_amb = total_tbat = 0
    for raw_path in raw_files:
        sid = raw_path.stem
        df = pd.read_parquet(raw_path)
        df_clean, n_amb, n_tbat = clean_temperatures(df)
        out_path = OUT_DIR / f"{sid}.parquet"
        safe_to_parquet(df_clean, out_path, index=False, compression="snappy")
        size_mb = out_path.stat().st_size / 1e6
        print(
            f"  [{sid}] {len(df_clean):,} rows, {size_mb:.1f} MB · "
            f"sentinels nulled: ambient={n_amb:,}, t_bat={n_tbat:,}",
            flush=True,
        )
        total_amb += n_amb
        total_tbat += n_tbat

    print(
        f"\ndone — total sentinels nulled: ambient={total_amb:,}, t_bat={total_tbat:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
