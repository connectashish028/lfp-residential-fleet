"""One-shot script to rebuild ``data/`` from the raw Figgener zips.

The ``data/`` directory is gitignored — the raw 1-second telemetry
zips total ~11 GB, far above GitHub's per-file limit. This script
runs the seven pipeline steps in order to rebuild bronze → silver →
gold from raw, assuming the zips are already in place locally.

Data source
-----------
Figgener, J. et al. (2024) "Multi-year field measurements of
home storage systems and their use in capacity estimation."
*Nature Energy* (open dataset).

Place the dataset zips at::

    data/raw/figgener_meta/Data_ID_*.zip
    data/raw/figgener_meta/Metadata/Systems_characterization.xlsx

Pipeline steps (each idempotent — re-run any step alone)
--------------------------------------------------------
1. ``raw_to_1min_parquet``      raw zips → 1-min parquet (bronze)
2. ``clean_temperatures``       sentinel scrub (-100 °C → NULL)
3. ``load_identity``            XLSX → identity.parquet
4. ``derive_features``          + ΔT, mode, energy_*, c_rate
5. ``derive_soc``               + OCV-corrected SoC
6. ``build_daily_kpis``         daily aggregates (gold)
7. ``detect_threshold_events``  rule-based events (gold)
8. ``degradation_modes``        ICA/DVA degradation-mode signatures (gold)

Total runtime: ~3-5 min on a modern laptop. Outputs ~700 MB total.

Run with::

    python bootstrap_data.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("data/raw/figgener_meta")

# Force UTF-8 in the child processes. The pipeline modules print Unicode
# (→, Δ, μ, °C); on a legacy Windows console (cp1252) an un-forced child
# raises UnicodeEncodeError mid-run. Linux/CI default to UTF-8 already.
_UTF8_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

PIPELINE_MODULES = [
    "bess_fleet.pipeline.raw_to_1min_parquet",
    "bess_fleet.pipeline.clean_temperatures",
    "bess_fleet.pipeline.load_identity",
    "bess_fleet.pipeline.derive_features",
    "bess_fleet.pipeline.derive_soc",
    "bess_fleet.pipeline.build_daily_kpis",
    "bess_fleet.pipeline.detect_threshold_events",
    "bess_fleet.pipeline.degradation_modes",
    "bess_fleet.pipeline.capacity_estimation",
]


def check_raw_present() -> bool:
    if not RAW_DIR.exists():
        return False
    return any(RAW_DIR.glob("Data_ID_*.zip"))


def main() -> None:
    if not check_raw_present():
        print(f"\n✖  Missing raw data at {RAW_DIR}")
        print("\nDownload the Figgener et al. 2024 dataset and place the zips at:")
        print(f"  {RAW_DIR.resolve()}")
        print("\nSee the module docstring for the citation and folder layout.")
        sys.exit(1)

    print(f"✓  Raw data found at {RAW_DIR}")
    print(f"   Running {len(PIPELINE_MODULES)} pipeline modules:\n")

    for i, module in enumerate(PIPELINE_MODULES, start=1):
        print(f"━━━ Step {i}/{len(PIPELINE_MODULES)}: {module} ━━━")
        result = subprocess.run([sys.executable, "-m", module], env=_UTF8_ENV)
        if result.returncode != 0:
            print(f"\n✖  Step {i} failed (exit code {result.returncode}).")
            sys.exit(result.returncode)
        print()

    print("✓  Done. Run `streamlit run app/Fleet_Overview.py` to start the dashboard.")


if __name__ == "__main__":
    main()
