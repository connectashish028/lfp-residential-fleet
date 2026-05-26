"""DuckDB query layer over the parquet lakehouse.

Mirrors the AWS Athena pattern: parquet files at rest, SQL on top. The
single ``data/bess.duckdb`` file is just a catalogue — data lives in
``data/lfp_1min/*.parquet`` and is read zero-copy. Delete the .duckdb
file any time and rerun ``build_views``; the underlying data is
unaffected.

Current scope (post-restructure 2026-05-16):

* ``telemetry_1min`` — six LFP systems at 1-minute cadence, no cleaning
  applied. Built by ``scripts/lfp_to_1min_parquet.py`` directly from the
  raw zips. Schema: timestamp, system_id, voltage_v, current_a, power_kw,
  temperature_c, ambient_c, interpolated_frac.

Usage::

    from bess_fleet.db import connect

    with connect() as con:
        df = con.sql(
            "SELECT system_id, AVG(temperature_c) "
            "FROM telemetry_1min WHERE system_id = 'ID14' "
            "GROUP BY 1"
        ).df()

The connection is read-only by default. Pass ``read_only=False`` only
from a build/maintenance script, never from notebooks or downstream
analysis code.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DUCKDB_PATH = DATA_DIR / "bess.duckdb"


VIEWS: dict[str, str] = {
    # Per-system parquet — one file per rack, globbed at query time. DuckDB
    # pushes predicates down to the file list, so WHERE system_id = 'ID14'
    # reads exactly one file out of six.
    #
    # telemetry_1min        — raw, preserves Figgener sentinels (-100 °C)
    # telemetry_1min_clean  — sentinels replaced with NULL, downstream-safe;
    #                         also carries derived columns (thermal_delta_c,
    #                         mode, is_idle, energy_*_step, c_rate)
    # identity              — per-system metadata (capacity, voltage, cells,
    #                         install date) from the Figgener XLSX
    "telemetry_1min":       "lfp_1min/*.parquet",
    "telemetry_1min_clean": "processed/*.parquet",
    "identity":             "identity.parquet",
    "daily_kpis":           "curated/daily_kpis.parquet",
    "threshold_events":     "curated/threshold_events.parquet",
}


def build_views(con: duckdb.DuckDBPyConnection) -> list[str]:
    """(Re-)register every parquet path in :data:`VIEWS` as a DuckDB view.

    Returns the list of view names that were registered. Missing files
    are silently dropped — that lets callers tolerate a partially-built
    lakehouse rather than crash.
    """
    registered: list[str] = []
    for name, rel in VIEWS.items():
        files = sorted(DATA_DIR.glob(rel))
        if not files:
            con.execute(f"DROP VIEW IF EXISTS {name}")
            continue
        glob_abs = (DATA_DIR / rel).as_posix()
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{glob_abs}')"
        )
        registered.append(name)
    return registered


@contextmanager
def connect(read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the project DuckDB and ensure views are up to date.

    Parameters
    ----------
    read_only:
        Default True. Set False only when the caller intends to mutate
        the catalogue (e.g. a build / maintenance script).
    """
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if read_only and not DUCKDB_PATH.exists():
        # DuckDB cannot open read-only on a non-existent file; create empty.
        duckdb.connect(str(DUCKDB_PATH)).close()
    con = duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
    try:
        if not read_only:
            build_views(con)
        yield con
    finally:
        con.close()


def list_views() -> list[str]:
    """Return the names of all currently-registered views."""
    with connect() as con:
        rows = con.sql("SHOW TABLES").fetchall()
    return [r[0] for r in rows]
