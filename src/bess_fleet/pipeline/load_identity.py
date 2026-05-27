"""Build the per-system identity table from the Figgener metadata XLSX.

Reads ``data/raw/figgener_meta/00_Data/00_Metadata/Metadata_Systems.xlsx``,
filters to the six LFP systems in scope, and writes
``data/identity.parquet`` with operator-friendly column names.

Columns written:

* ``system_id``         — e.g. ``"ID14"``
* ``manufacturer``      — vendor name
* ``chemistry``         — high-level cell chemistry (LFP)
* ``chemistry_detail``  — sub-variant if specified
* ``capacity_kwh``      — nominal energy capacity
* ``capacity_ah``       — nominal Ah capacity (needed for c-rate)
* ``voltage_nominal_v`` — pack nominal voltage
* ``cells_series``      — number of cells in series
* ``cells_parallel``    — number of cells in parallel
* ``install_date``      — when the system was commissioned

Run with::

    python -m bess_fleet.pipeline.load_identity
"""

from __future__ import annotations

import pandas as pd

from bess_fleet.db import DATA_DIR
from bess_fleet.io import safe_to_parquet

META_XLSX = (
    DATA_DIR / "raw" / "figgener_meta" / "00_Data" / "00_Metadata"
    / "Metadata_Systems.xlsx"
)
OUT_PATH = DATA_DIR / "identity.parquet"

# Active scope — the 6 LFP systems whose 1-min parquets live in data/lfp_1min/.
LFP_IDS: frozenset[str] = frozenset(
    {"ID14", "ID16", "ID17", "ID18", "ID19", "ID20"}
)


def main() -> None:
    if not META_XLSX.exists():
        raise SystemExit(f"missing metadata XLSX: {META_XLSX}")

    meta = pd.read_excel(META_XLSX, sheet_name="Metadata")
    meta = meta.assign(
        system_id=meta["ID"].astype(int).map(lambda i: f"ID{i:02d}"),
        install_date=pd.to_datetime(
            meta["Date_storage_system_installation"], unit="s"
        ),
    )

    keep_cols = [
        "system_id", "Manufacturer", "Chemistry", "Chemistry_detail",
        "Energy_nominal_in_kWh", "Capacity_nominal_in_Ah",
        "Voltage_nominal_in_V", "Cell_number_in_series",
        "Cell_number_in_parallel", "install_date",
    ]
    rename = {
        "Manufacturer":            "manufacturer",
        "Chemistry":               "chemistry",
        "Chemistry_detail":        "chemistry_detail",
        "Energy_nominal_in_kWh":   "capacity_kwh",
        "Capacity_nominal_in_Ah":  "capacity_ah",
        "Voltage_nominal_in_V":    "voltage_nominal_v",
        "Cell_number_in_series":   "cells_series",
        "Cell_number_in_parallel": "cells_parallel",
    }
    identity = (
        meta[meta["system_id"].isin(LFP_IDS)][keep_cols]
        .rename(columns=rename)
        .sort_values("system_id")
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_to_parquet(identity, OUT_PATH, index=False, compression="snappy")
    print(f"wrote {OUT_PATH}: {len(identity)} rows\n", flush=True)
    print(identity.to_string(index=False))


if __name__ == "__main__":
    main()
