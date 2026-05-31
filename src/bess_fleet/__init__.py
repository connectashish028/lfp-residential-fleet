"""BESS fleet health — cross-chemistry battery-diagnostic platform.

A medallion-lakehouse pipeline (raw zips → bronze 1-min → silver →
gold) over parquet + DuckDB, feeding a Streamlit dashboard. The bronze
builder is :mod:`bess_fleet.pipeline.raw_to_1min_parquet`; the query
layer is :mod:`bess_fleet.db`.
"""

from bess_fleet import db

__all__ = ["db"]
__version__ = "0.2.0"
