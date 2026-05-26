"""BESS fleet health — minimal scaffold.

Current scope: raw → 1-minute parquet conversion (see
``scripts/lfp_to_1min_parquet.py``) and a DuckDB view layer over the
result (see :mod:`bess_fleet.db`). Everything else has been removed and
will be rebuilt step by step.
"""

from bess_fleet import db

__all__ = ["db"]
__version__ = "0.2.0"
