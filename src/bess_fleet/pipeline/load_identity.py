"""Build the per-system identity table from the Figgener metadata XLSX.

Reads ``data/raw/figgener_meta/00_Data/00_Metadata/Metadata_Systems.xlsx``,
filters to the systems in scope (cross-chemistry), and writes
``data/identity.parquet`` with operator-friendly column names. The
``chemistry`` column it carries is what lets every downstream stage —
SoC OCV-table selection, degradation-mode signatures — branch on
chemistry instead of assuming LFP.

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
* ``inverter_power_kw`` — nominal inverter power (sets the C-rate ceiling)
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

# Active scope — must match SYSTEM_IDS in raw_to_1min_parquet.py, grouped
# by chemistry so the identity table carries the cross-chemistry mix.
LFP_E:     frozenset[str] = frozenset({"ID14", "ID16", "ID17", "ID18", "ID19", "ID20"})  # Mfr E · LFP
LMO_NMC_A: frozenset[str] = frozenset({"ID01", "ID02"})                                  # Mfr A · LMO/NMC blend
NMC_BC:    frozenset[str] = frozenset({"ID07", "ID11"})                                  # Mfr B/C · pure NMC
SYSTEM_IDS: frozenset[str] = LFP_E | LMO_NMC_A | NMC_BC


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
        "Cell_number_in_parallel", "Inverter_nominal_power", "install_date",
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
        "Inverter_nominal_power":  "inverter_power_kw",
    }
    identity = (
        meta[meta["system_id"].isin(SYSTEM_IDS)][keep_cols]
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
