"""Build-pipeline scripts for the BESS Fleet Health project.

Each module is idempotent and runnable as ``python -m
bess_fleet.pipeline.<name>``. Run order (each step reads the
upstream layer and writes its own):

1. ``raw_to_1min_parquet``      — raw zips → 1-min parquet (bronze)
2. ``clean_temperatures``       — sentinel scrub (-100 °C → NULL)
3. ``load_identity``            — XLSX metadata → identity.parquet
4. ``derive_features``          — ΔT, mode, energy_*, c_rate
5. ``derive_soc``               — chemistry-aware OCV-corrected SoC
6. ``build_daily_kpis``         — daily aggregates with RTE confidence gate
7. ``detect_threshold_events``  — chemistry-aware rule-based event log
8. ``degradation_modes``        — ICA/DVA degradation modes (LLI vs LAM)

Each module exposes a ``main()`` for direct invocation and the pure
functions it uses for unit testing.
"""
from __future__ import annotations

import contextlib
import sys


def _configure_console() -> None:
    """Force UTF-8 on stdout/stderr so the pipeline's ``→ / Δ / μ / °C``
    prints don't raise ``UnicodeEncodeError`` on a legacy Windows (cp1252)
    console when a module is run directly via ``python -m
    bess_fleet.pipeline.<name>``. (``bootstrap_data.py`` sets the same via
    the child env; this covers direct invocations.) Guarded — a no-op
    where the stream has no ``reconfigure`` or it can't be changed.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


_configure_console()
