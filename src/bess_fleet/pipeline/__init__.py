"""Build-pipeline scripts for the BESS Fleet Health project.

Each module is idempotent and runnable as ``python -m
bess_fleet.pipeline.<name>``. Run order (each step reads the
upstream layer and writes its own):

1. ``lfp_to_1min_parquet`` — raw zips → 1-min parquet (bronze)
2. ``clean_temperatures``  — sentinel scrub (-100 °C → NULL)
3. ``load_identity``       — XLSX metadata → identity.parquet
4. ``derive_features``     — ΔT, mode, energy_*, c_rate
5. ``derive_soc``          — OCV-corrected coulomb-counted SoC
6. ``build_daily_kpis``    — daily aggregates with RTE confidence gate
7. ``detect_threshold_events`` — rule-based event log

Each module exposes a ``main()`` function for direct invocation and
the pure functions it uses for unit testing.
"""
