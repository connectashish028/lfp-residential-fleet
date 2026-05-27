"""Add derived columns to the cleaned per-system parquets.

Reads and overwrites ``data/processed/*.parquet`` in place. Designed to
grow over time — add new derived features here, re-run, and every
downstream consumer picks them up automatically through the
``telemetry_1min_clean`` DuckDB view.

Currently derives:

* ``thermal_delta_c``       — ``temperature_c − ambient_c``. The
  rack-internal heat residual. NaN-safe.
* ``mode``                  — ``'charge' / 'discharge' / 'idle'`` from
  ``power_kw`` with ±0.05 kW deadband. The single most useful
  categorical for filter + groupby work.
* ``is_idle``               — ``mode == 'idle'`` boolean. Fast filter.
* ``energy_kwh_step``       — ``power_kw / 60``. Per-minute energy,
  signed (positive = charge, negative = discharge).
* ``energy_in_kwh_step``    — ``energy_kwh_step.clip(lower=0)``. The
  denominator of round-trip efficiency.
* ``energy_out_kwh_step``   — ``(-energy_kwh_step).clip(lower=0)``. The
  numerator of round-trip efficiency, as a positive value.
* ``c_rate``                — ``|current_a| / capacity_ah``. Dimensionless
  duty intensity. Requires ``data/identity.parquet``.

Run order (each script is idempotent; re-run any step to refresh):

    1. python -m bess_fleet.pipeline.lfp_to_1min_parquet   # raw zip → data/lfp_1min/
    2. python -m bess_fleet.pipeline.clean_temperatures    # data/lfp_1min/ → data/processed/
    3. python -m bess_fleet.pipeline.load_identity         # XLSX → data/identity.parquet
    4. python -m bess_fleet.pipeline.derive_features       # adds derived cols to data/processed/

Step 4 modifies ``data/processed/`` in place. If you re-run step 2,
re-run step 4 to restore derived columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_fleet.db import DATA_DIR
from bess_fleet.io import safe_to_parquet

PROCESSED_DIR = DATA_DIR / "processed"
IDENTITY_PATH = DATA_DIR / "identity.parquet"

# 1-minute cadence — Δt in hours for energy = power × Δt
DT_HOURS = 1.0 / 60.0
# Power deadband for mode classification (±50 W)
MODE_DEADBAND_KW = 0.05


def _load_capacity_lookup() -> dict[str, float]:
    """Return system_id → capacity_ah lookup. Required for c_rate."""
    if not IDENTITY_PATH.exists():
        raise SystemExit(
            f"missing {IDENTITY_PATH}. Run `python -m bess_fleet.pipeline.load_identity` first."
        )
    ident = pd.read_parquet(IDENTITY_PATH)
    return dict(zip(ident["system_id"], ident["capacity_ah"], strict=True))


def derive_features(df: pd.DataFrame, capacity_ah: float) -> pd.DataFrame:
    """Apply all derived features. Idempotent — re-running replaces values."""
    out = df.copy()

    # Thermal residual — rack-internal heat. NaN-safe.
    out["thermal_delta_c"] = out["temperature_c"] - out["ambient_c"]

    # Dispatch mode + idle boolean
    mode_arr = np.select(
        [out["power_kw"] > MODE_DEADBAND_KW, out["power_kw"] < -MODE_DEADBAND_KW],
        ["charge", "discharge"],
        default="idle",
    )
    out["mode"] = pd.Series(mode_arr, index=out.index, dtype="string")
    out["is_idle"] = out["mode"] == "idle"

    # Per-minute energy — signed (charge +, discharge −) and clipped variants
    out["energy_kwh_step"] = (out["power_kw"] * DT_HOURS).astype("float32")
    out["energy_in_kwh_step"]  = out["energy_kwh_step"].clip(lower=0)
    out["energy_out_kwh_step"] = (-out["energy_kwh_step"]).clip(lower=0)

    # C-rate — dimensionless duty intensity
    out["c_rate"] = (out["current_a"].abs() / capacity_ah).astype("float32")

    return out


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"no parquets in {PROCESSED_DIR}. Run clean_temperatures.py first."
        )
    capacity_lookup = _load_capacity_lookup()
    print(
        f"deriving features for {len(files)} parquets in {PROCESSED_DIR}\n",
        flush=True,
    )

    for path in files:
        sid = path.stem
        if sid not in capacity_lookup:
            print(f"  [{sid}] SKIP — no capacity_ah in identity table", flush=True)
            continue
        df = pd.read_parquet(path)
        out = derive_features(df, capacity_ah=float(capacity_lookup[sid]))
        safe_to_parquet(out, path, index=False, compression="snappy")

        # Per-system stats — sanity check the new columns
        dt = out["thermal_delta_c"]
        mode_counts = out["mode"].value_counts()
        idle_pct = mode_counts.get("idle", 0) / len(out) * 100
        mean_c = float(out["c_rate"].mean())
        print(
            f"  [{sid}] ΔT μ={dt.mean():.2f}°C  "
            f"idle={idle_pct:.1f}%  c̄_rate={mean_c:.3f}  "
            f"rows={len(out):,}",
            flush=True,
        )

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
