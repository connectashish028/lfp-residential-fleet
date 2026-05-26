"""Safe file I/O utilities for the pipeline layer.

The pipeline scripts write derived parquets in place. A direct
``df.to_parquet(path, ...)`` is *not* atomic — PyArrow's writer
emits header, row groups, then footer; if the process is killed
before the footer flushes, the target file is unreadable and the
next pipeline run crashes on read.

:func:`safe_to_parquet` makes the write atomic via a temp-file +
rename pattern. The target file either has the previous content or
the new content — never partial.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def safe_to_parquet(
    df: pd.DataFrame,
    path: str | Path,
    **to_parquet_kwargs: Any,
) -> None:
    """Atomically write a DataFrame to parquet.

    The DataFrame is written to ``<path>.tmp`` first; on success the
    temp file is atomically renamed to ``<path>`` via :func:`os.replace`.
    POSIX guarantees rename atomicity (cross-platform from Python 3.3+),
    so a crash mid-write leaves either the old file intact or the new
    file in place — never a half-written parquet.

    Parameters
    ----------
    df
        The DataFrame to persist.
    path
        Target parquet path. The temp file lives next to it as
        ``<path>.tmp``.
    **to_parquet_kwargs
        Forwarded to :meth:`pandas.DataFrame.to_parquet`. Typical
        usage: ``index=False, compression="snappy"``.

    Notes
    -----
    The temp file is cleaned up by the rename. If the underlying
    ``to_parquet`` call raises, the temp file is left in place so
    the pipeline operator can inspect it; it does **not** clobber
    the target.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, **to_parquet_kwargs)
    os.replace(tmp, path)
