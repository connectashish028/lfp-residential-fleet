# LFP residential fleet — health & analytics dashboard

A TWAICE-style operator dashboard for residential battery energy
storage, built on the [Figgener et al. 2024](https://www.nature.com/articles/s41597-024-02944-7)
open dataset of six LFP racks instrumented at one-minute cadence from
2015 through 2022.

The repo is organised as a proper Python package: a medallion-lakehouse
pipeline over DuckDB feeds an analytics layer; a Streamlit app sits on
top and reads only through a single-source-of-truth module.

---

## What's in the box

```
src/bess_fleet/
├── db.py                       # DuckDB view registrar
├── recommendations.py          # operator-facing rule engine
└── pipeline/                   # 7 idempotent build modules
    ├── lfp_to_1min_parquet.py    bronze: raw zips → 1-min parquet
    ├── clean_temperatures.py     silver: sentinel scrub
    ├── load_identity.py          silver: XLSX → identity.parquet
    ├── derive_features.py        silver: ΔT, c_rate, energy_*
    ├── derive_soc.py             silver: OCV-corrected SoC
    ├── build_daily_kpis.py       gold:   daily aggregates + 4-gate RTE
    └── detect_threshold_events.py gold:  rule-based events

app/
├── Fleet_Overview.py           # severity-first systems table
├── pages/1_System.py           # per-rack telemetry deep-dive
└── _components/
    ├── data_access.py           cached DuckDB queries (`get_*`)
    ├── analytics.py             cached compute functions (`compute_*`)
    ├── data.py                  re-export facade for the above
    ├── charts.py                Plotly chart builders
    ├── kpis.py                  layout primitives (HTML)
    ├── alerts.py                alert-detail UI helpers
    └── theme.py                 Operator-Light CSS

tests/                          # 76 tests, runs in < 2 s
docs/                           # interview-prep walkthrough + Q&A
Notebooks/                      # audit notebook (SoC stress tests)
```

---

## Headline KPIs

All four are computed in [`build_daily_kpis.py`](src/bess_fleet/pipeline/build_daily_kpis.py)
or in the analytics layer:

| KPI | Definition | Notes |
|---|---|---|
| **Daily RTE** | `Σ energy_out / Σ energy_in` per day | Four-condition gate — see below |
| **Daily cycling (EFC/day)** | `throughput / (2 × nameplate)` | Capacity-relative throughout |
| **State of Health** | OCV-anchored coulomb counting, monthly median, normalised to first-six-month baseline | Plett 2015 ch. 8 hybrid; clipped to [70, 100] |
| **Availability** | `n_samples × (1 − interp_frac) / 1440`, capped 100 % | DST-cap + interpolation-discount |

### The RTE confidence gate

`RTE` returns `NULL` unless **all four** are true. Capacity-relative
thresholds generalise the rule across fleet sizes (5 kWh → 1 MWh)
without re-tuning:

```sql
energy_in  >= 0.10 × capacity_kwh    -- meaningful charging
energy_out >= 0.05 × capacity_kwh    -- meaningful discharging
energy_out / energy_in <= 1.05       -- physically plausible
ABS(soc_end - soc_start) <= 10       -- cycle closes
```

After the gate, 38–83 % of daily rows return NULL across the fleet —
by design. The survivors give a fleet mean RTE of 81–88 %, the LFP
residential ballpark.

---

## Running it

```bash
# 1. install
pip install -e .[dev]

# 2. fetch the Figgener dataset (citation in bootstrap_data.py),
#    place zips at data/raw/figgener_meta/Data_ID_*.zip

# 3. rebuild bronze → silver → gold (~3 min)
python bootstrap_data.py

# 4. tests
pytest tests/                # 76 tests in < 2 s

# 5. dashboard
streamlit run app/Fleet_Overview.py
```

The `data/` folder is git-ignored (the raw zips total ~12 GB) — see
[`bootstrap_data.py`](bootstrap_data.py) for the data source and the
pipeline order.

---

## Architectural decisions

- **Parquet on disk = source of truth.** DuckDB is the query layer
  over parquet globs; the catalogue file (`data/bess.duckdb`) is
  expendable and rebuilds in ~200 ms from any state.
- **Single source of truth in `data.py`.** Every numeric quantity on
  the UI is computed in one module and consumed by every page. Charts
  do visuals; pages compose. No SQL in page code.
- **Idempotent pipeline.** Each of the seven build modules can be re-
  run alone to refresh its output without recomputing the upstream
  layer.
- **Capacity-relative thresholds.** Anywhere a threshold could be
  expressed as a fraction of nameplate (RTE confidence gates, EFC,
  C-rate), it is. The same code is correct on a 5 kWh residential
  rack and a 1 MWh utility system.

---

## Methodology

For the per-KPI rationale, the data-quality caveats per system
(ID19/ID20 plateau effect, ID18 short tenure), and the headline
ID17 finding (elevated internal resistance via the DoD-vs-RTE slope),
see [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).

For the eight SoC-algorithm stress tests and the slope-fit numbers,
see [`Notebooks/daily_kpis_eda.ipynb`](Notebooks/daily_kpis_eda.ipynb).

---

## Data

[Figgener, J. et al. (2024)](https://doi.org/10.1038/s41597-024-02944-7).
Multi-year field measurements of home storage systems and their use
in capacity estimation. *Scientific Data*, open dataset.

Six LFP residential systems, ~12 M rows of 1-minute telemetry across
2015–2022, single manufacturer.

---

## Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Storage | Apache Parquet (PyArrow 16) |
| Query | DuckDB 1.5 |
| Data | pandas 2.2, numpy 1.26 |
| UI | Streamlit 1.35, Plotly 5.20 |
| Tests | pytest 8 |
