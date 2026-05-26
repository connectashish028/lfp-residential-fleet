# BESS Fleet Health — Project Summary

A fleet health dashboard for residential battery storage, built on
the Figgener et al. 2024 open dataset. Operator-facing severity-first
UI; engineer-auditable methodology behind every number.

---

## What this project does

- Six residential LFP systems, 2015 → 2022, 1-minute cadence, single manufacturer (Figgener Mfr E).
- Two complementary readings on every page:
  - **5-second glance** (operator): which system needs attention today, and what's the next action?
  - **5-minute audit** (engineer): every KPI definition, every algorithm caveat, every slope value is auditable through the analytics module and the included notebook.
- Three analytical pillars: two-tier alerting (threshold + statistical), peer comparison on hardware-identical units, severity-ranked actionable recommendations.

---

## Data architecture — medallion lakehouse over DuckDB

- **Raw** — `data/raw/` — 12 immutable zips (1-second telemetry)
- **Bronze** — `data/lfp_1min/*.parquet` — 1-min downsampled, 6 files
- **Silver** — `data/processed/*.parquet` — sentinel-scrubbed + derived columns
- **Gold** — `data/curated/{daily_kpis,threshold_events}.parquet` — analytical layer
- **Catalogue** — `data/bess.duckdb` — five DuckDB views over the parquet glob; zero-copy reads, deletable any time and rebuildable from parquet

---

## Build pipeline (each script idempotent — re-run any step)

1. `lfp_to_1min_parquet.py` — raw zips → 1-min parquet
2. `clean_temperatures.py` — Figgener `-100 °C` sentinel scrub → NaN
3. `load_identity.py` — XLSX metadata → `identity.parquet`
4. `derive_features.py` — ΔT, mode, is_idle, energy_*, c_rate
5. `derive_soc.py` — OCV-corrected coulomb-counted SoC + anchor flags
6. `build_daily_kpis.py` — daily aggregates with confidence gating
7. `detect_threshold_events.py` — rule-based event log

---

## KPI design

### Round-trip efficiency (RTE)
- **Formula**: `Σ energy_out_kwh / Σ energy_in_kwh` per day
- **Confidence gate** — daily RTE returns `NULL` unless all three:
  - `energy_in ≥ 1.0 kWh` (meaningful charging)
  - `energy_out ≥ 0.5 kWh` (meaningful discharging)
  - ratio `≤ 1.05` (physically plausible)
- **Window**: 30-day median, relative to each rack's own last sample

### Daily Cycling (EFC / day)
- **Formula**: `daily throughput_kwh / (2 × nameplate_kwh)`
- **Window**: 30-day median
- Residential racks typically run 0.3 – 0.8 EFC/day

### State of Health (SoH, derived)
- **Method** — Plett 2015 ch. 8 (OCV + CC), with per-system commissioning-baseline anchoring:
  - Anchor pairs must bridge both OCV cliffs (V < 3.10 or V > 3.36)
  - `|ΔSoC_ocv| ≥ 30 %`, elapsed ≤ 24 h, sign(ΔAh) = sign(ΔSoC)
  - `implied_capacity_Ah = ΔAh / (ΔSoC / 100)`
  - Monthly median → divide by **median of first 6 months** → SoH %, clipped [70, 100]
  - Min 3 qualifying months or the series is omitted
- **Tier 1** (ID14/16/17/18) — anchors distribute across cliffs; absolute SoH trustworthy
- **Tier 2** (ID19/20) — >90 % anchors in plateau; no qualifying pairs, SoH omitted (data limitation, not algorithm bug)

### Availability
- **Formula**: `n_samples × (1 − mean(interpolated_frac)) / 1440`, capped at 100 %
- Two corrections vs raw row-count:
  - **DST cap** — autumn 25-hour days don't leak above 100 %
  - **Interpolation discount** — Figgener-reconstructed minutes weighted down so the metric reflects real data presence
- 90 % outage threshold reference line on the chart; red dots for outage days

### DoD-vs-RTE slope (the killer diagnostic)
- Per-rack linear fit of daily RTE % against daily DoD %
- **Flat slope = healthy**; **steep negative slope = I²R fingerprint of elevated internal resistance**
- **Single source of truth**: `data.compute_dod_vs_rte_fits()` — used by Overview callout, Peer Comparison chart titles, Group statistics table

### System status pill (Overview)
- `critical_events > 0` → **CRITICAL**
- `warning_events > 0` OR `mean_dt_c > 5 °C` OR notable finding → **WATCH**
- `last_seen` older than 60 days → **RETIRED**
- else → **HEALTHY**

---

## Pages

### 1. Fleet Overview (`app/Fleet_Overview.py`)
- **Historical-replay banner** — data window `2015-07 → 2022-12`, "Now anchor" = dataset's last sample
- **System status table** — five columns (System / Performance Status / Safety Index / State of Health / Status) with severity-coloured dots. Retirement-aware: retired racks show `last seen YYYY-MM`. Notable-finding override surfaces the ID17 *Watch* pill.

### 2. System — deep dive (`app/pages/1_System.py`)
- **System + Time Window selectors** (top row) — Time Window cycles 30 / 45 / 90 / 365 days and drives every chart on the page.
- **Identity strip** — capacity, nominal voltage, cells in series, install date.
- **Usable & Recoverable Energy** — stacked-bar daily breakdown (Usable / Cycle Loss / Missing data) alongside a donut of the period totals plus an *Aging (est.)* slice from the SoH summary.
- **Availability chart** — daily availability bars with 90 % threshold reference line.
- **Daily RTE chart** — severity-coloured bars driven by the four-condition confidence gate; days that fail the gate are nulled out.
- **Daily Cycling chart** — daily EFC/day, blue bars.
- **Telemetry tabs** — SoC / Thermal / Dispatch:
  - Each tab carries a timeline chart, an inline KPI badge, a Min/Avg/Max stat row, and a KDE distribution split by resting vs operating state.
  - Click a marker on the timeline → alert-detail callout with the operator recommendation.
- **Alerts table** — top 5 most-recent threshold events with severity pill, peak / duration, action sentence. Each row opens a modal showing the day-of-event telemetry slice with the rule's threshold band overlaid.

---

## Headline analytical finding

- **ID17 — elevated internal resistance, capacity preserved**
- DoD-vs-RTE slope: **−0.350 pp / % DoD** (17 × steeper than ID14, 1 × steeper than ID16)
- SoH: ~100 % (no measurable capacity loss)
- Mean ΔT spike during dispatch was a **sensor-placement issue**, not real thermal stress
- **Recommendation**: load test on next O&M visit; if confirmed, candidate for early replacement
- Talking point: *"capacity and impedance fail on independent axes — a real fleet-health platform needs both"*

---

## Analytical pillars

| Pillar | Implementation |
|---|---|
| Two-tier alerting (thresholds + statistical anomalies) | `detect_threshold_events.py` + within-system robust-z (notebook) |
| Peer comparison on hardware-identical units | DoD-vs-RTE slope within 8 kWh / 9 kWh groups (notebook) |
| Severity-ranked actionable insights | `recommendations.py` — every alert carries an operator runbook line |

---

## Recommendation engine (`src/bess_fleet/recommendations.py`)

- Pure Python — no DuckDB, no Streamlit, no pandas
- Static `rule_id → (severity, action_template)` lookup
- Voice rules:
  - Short, full-stop sentences
  - Severity-appropriate verb at the front (Verify · Inspect · Limit · Isolate · Dispatch)
  - Always ends in a concrete action
  - No hedging
- Three callable surfaces:
  - `for_threshold_event(rule_id, peak, duration)` — wired into the alerts table + chart marker popups
  - `for_rte_drop(magnitude_pp)` — for daily-KPI anomalies
  - `for_high_dt(mean_dt_c)` — for thermal anomalies

---

## Stress-testing & validation

- **SoC algorithm** — eight stress tests on all six systems (notebook `daily_kpis_eda.ipynb` § 8)
  - Anchor self-consistency: RMSE = 0 by construction
  - Dispatch monotonicity: ≥ 98.6 % pass rate fleet-wide
  - Daily DoD vs |throughput Ah| r²: 0.70 – 0.84
- **Availability metric** — DST-cap + interpolation-discount audited against raw row counts
- **DoD-vs-RTE slope** — single-source-of-truth function used by every UI surface; no number can drift

---

## Design language — Operator-Light theme

- Light variant of the project's Operator Dark spec (`theme.md`)
- **Two-accent rule** — lilac `#6B5BCC` for derived / model values, blue `#2563EB` for measured / realised values; severity palette (amber / red / green) reserved for status pills
- **Typography** — Inter for prose, JetBrains Mono for every numeric / metadata, uppercase letter-spaced labels
- No rounded corners, no shadows, no gradients
- ⓘ hover tooltips on every KPI tile and chart heading instead of verbose paragraph prose
- `(derived)` badge next to KPIs that come from a computation rather than a raw measurement

---

## Stack

| Layer | Library / version |
|---|---|
| Language | Python 3.11+ |
| Storage | Apache Parquet via PyArrow 16 |
| Query | DuckDB 1.5 (analytical SQL over parquet glob) |
| Data | pandas 2.2 + numpy 1.26 |
| UI | Streamlit 1.57 |
| Charts | Plotly 5.24 |
| Identity I/O | openpyxl 3.1 (Figgener metadata XLSX) |
