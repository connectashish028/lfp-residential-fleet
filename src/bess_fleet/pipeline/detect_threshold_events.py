"""Threshold-based event detection on telemetry_1min_clean.

Complements the within-system statistical detector. Where statistical
methods catch *deviations from normal*, threshold rules catch *hard rule
violations* — the kind of thing an operator must act on immediately.

Rules are organised by physics domain and audited against the actual
fleet distribution (see WORK_LOG.md). Each rule has a threshold,
severity tier, minimum sustained duration, and optionally a system
filter (for capacity-stratified rules — e.g. ΔT thresholds differ
between the 8 kWh and 9.2 kWh hardware groups).

Output: ``data/curated/threshold_events.parquet`` — one row per event.

Schema:

* ``system_id``         — e.g. ``"ID14"``
* ``rule_id``           — short rule identifier
* ``severity``          — ``warning`` / ``critical``
* ``channel``           — telemetry column that fired the rule
* ``start``, ``end``    — event boundaries (1-min cadence)
* ``duration_min``      — sustained duration
* ``peak_value``        — |value|.max() during the event

Run with::

    python -m bess_fleet.pipeline.detect_threshold_events
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from bess_fleet.db import DATA_DIR, connect
from bess_fleet.io import safe_to_parquet

OUT_PATH = DATA_DIR / "curated" / "threshold_events.parquet"

# Hardware groups — different capacity → different thermal profile,
# stratified ΔT thresholds reflect this.
PACK_8KWH: frozenset[str] = frozenset({"ID14", "ID16", "ID17", "ID18"})
PACK_9KWH: frozenset[str] = frozenset({"ID19", "ID20"})

# A C-rate above (inverter rated C-rate × this margin) is "above inverter".
# inverter rated C-rate = inverter_kw / capacity_kwh = I/Q (since P=VI,
# E=VQ), so it's directly comparable to the measured c_rate channel.
C_RATE_MARGIN = 1.4

# ─── RULES — audited against fleet stats ───────────────────────────────────
# Each rule: (rule_id, severity, channel, condition_fn, min_duration_min, system_filter)
RULES: list[tuple[str, str, str, Callable[[pd.Series], pd.Series], int, frozenset[str] | None]] = [
    # ─ Thermal ──────────────────────────────────────────────────────────
    # T_bat > 45 fires on ID19/ID20 (hotter 9 kWh group). Sub-critical.
    ("t_bat_warm",         "warning",  "temperature_c",
        lambda s: s > 45,                              5,  None),
    # T_bat > 50 — safety rule, currently dormant on the fleet
    ("t_bat_high",         "warning",  "temperature_c",
        lambda s: s > 50,                              5,  None),
    # T_bat > 60 — thermal-runaway risk, dormant
    ("t_bat_critical",     "critical", "temperature_c",
        lambda s: s > 60,                              1,  None),
    # T_bat < 0 — cold-charge risk (lithium plating in LFP)
    ("t_bat_cold",         "warning",  "temperature_c",
        lambda s: s < 0,                              15,  None),
    # ΔT — capacity-stratified: 9 kWh runs hotter by design
    ("delta_t_high_8kwh",  "warning",  "thermal_delta_c",
        lambda s: s.abs() > 10,                       15,  PACK_8KWH),
    ("delta_t_high_9kwh",  "warning",  "thermal_delta_c",
        lambda s: s.abs() > 15,                       15,  PACK_9KWH),

    # ─ Electrical ───────────────────────────────────────────────────────
    # c_rate > 1.0 — physically impossible on any of these packs; a clear
    # measurement error. The softer "above the inverter's rated C-rate"
    # check is inverter-relative and lives in main() (see C_RATE_MARGIN):
    # a fixed 0.6 ceiling is right for an 8 kWh / 3.3 kW LFP rack (~0.41 C)
    # but fires on every normal sample of a 2.2 kWh / 2.0 kW unit (~0.91 C).
    ("c_rate_impossible",     "critical", "c_rate",
        lambda s: s > 1.0,                             1,  None),
    # Cell-voltage rules — derived as voltage_v / cells_series. These are
    # chemistry-specific (see _RULE_CHEMISTRIES): a 3.65 V "overcharge"
    # limit is correct for LFP but would fire on every normal NMC sample
    # (~4.2 V full charge). main() gates each rule to its chemistry.
    #
    # ── LFP (full charge ~3.65 V) ────────────────────────────────────────
    ("cell_v_overcharge",     "critical", "cell_voltage_v",
        lambda s: s > 3.65,                            1,  None),
    # Two-tier undervolt — separates "BMS working hard at cutoff" from
    # "BMS failing to protect":
    #
    #   cell_v_low:    0.5 < v < 2.5  → BMS hit cutoff; cell is marginal
    #                  but the BMS is doing its job. Warning, not critical.
    #   cell_v_deep:   0.5 < v < 2.0  → cell sustained well below safe
    #                  range. BMS failed; cell-damage territory.
    ("cell_v_low",            "warning",  "cell_voltage_v",
        lambda s: (s > 0.5) & (s < 2.5),               5,  None),
    ("cell_v_deep_undervolt", "critical", "cell_voltage_v",
        lambda s: (s > 0.5) & (s < 2.0),               5,  None),
    # ── NMC / LMO-NMC (full charge ~4.2 V) ───────────────────────────────
    # Conservative limits — clearly-abnormal only, so they stay dormant on
    # healthy cells rather than risk a false-positive flood pending a
    # proper per-cell audit.
    ("nmc_cell_v_overcharge", "critical", "cell_voltage_v",
        lambda s: s > 4.25,                            1,  None),
    ("nmc_cell_v_undervolt",  "warning",  "cell_voltage_v",
        lambda s: (s > 0.5) & (s < 2.5),               5,  None),
    ("nmc_cell_v_deep",       "critical", "cell_voltage_v",
        lambda s: (s > 0.5) & (s < 2.0),               5,  None),

    # ─ Operational ──────────────────────────────────────────────────────
    # System dark: cell_v < 0.5 for ≥30 min → system off / comms outage.
    # Chemistry-agnostic (0.5 V/cell means "off" on any chemistry).
    ("system_dark",           "warning",  "cell_voltage_v",
        lambda s: s < 0.5,                            30,  None),
]

# Cell-voltage rules are calibrated per chemistry; every other rule
# (thermal, c-rate, system_dark) is physical and applies to all. A rule
# absent from this map runs on every chemistry.
_RULE_CHEMISTRIES: dict[str, frozenset[str]] = {
    "cell_v_overcharge":      frozenset({"LFP"}),
    "cell_v_low":             frozenset({"LFP"}),
    "cell_v_deep_undervolt":  frozenset({"LFP"}),
    "nmc_cell_v_overcharge":  frozenset({"NMC", "LMO"}),
    "nmc_cell_v_undervolt":   frozenset({"NMC", "LMO"}),
    "nmc_cell_v_deep":        frozenset({"NMC", "LMO"}),
}


def collapse_to_events(
    flag_series: pd.Series,
    value_series: pd.Series,
    min_duration_min: int = 1,
) -> pd.DataFrame:
    """Run-length collapse: consecutive flag rows → one event row."""
    f = np.asarray(flag_series.fillna(False).astype(int).values)
    if f.sum() == 0:
        return pd.DataFrame(columns=["start", "end", "duration_min", "peak_value"])
    bound = np.diff(np.concatenate([[0], f, [0]]))
    starts_idx = np.where(bound == 1)[0]
    ends_idx = np.where(bound == -1)[0] - 1
    idx = flag_series.index
    rows: list[dict[str, object]] = []
    for s, e in zip(starts_idx, ends_idx, strict=True):
        dur = (idx[e] - idx[s]).total_seconds() / 60.0 + 1.0
        if dur < min_duration_min:
            continue
        peak = float(value_series.iloc[s:e + 1].abs().max())
        rows.append({
            "start": idx[s],
            "end": idx[e],
            "duration_min": dur,
            "peak_value": peak,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"writing to {OUT_PATH}\n", flush=True)

    # Pull identity once for cell-count lookup (used in cell_voltage derivation)
    with connect() as con:
        identity = con.sql(
            "SELECT system_id, cells_series, chemistry, "
            "inverter_power_kw, capacity_kwh FROM identity ORDER BY system_id"
        ).df()
    systems = identity["system_id"].tolist()
    cells_lookup = dict(zip(identity["system_id"], identity["cells_series"], strict=True))
    chem_lookup = dict(zip(identity["system_id"], identity["chemistry"], strict=True))
    # Inverter rated C-rate per system = inverter_kw / capacity_kwh.
    ceiling_lookup = {
        r["system_id"]: float(r["inverter_power_kw"]) / float(r["capacity_kwh"])
        for _, r in identity.iterrows()
        if r["capacity_kwh"]
    }

    all_events: list[pd.DataFrame] = []
    for sid in systems:
        with connect() as con:
            one = con.sql(f"""
                SELECT timestamp, temperature_c, ambient_c, thermal_delta_c,
                       voltage_v, current_a, power_kw, c_rate, mode, is_idle
                FROM telemetry_1min_clean
                WHERE system_id = '{sid}'
                ORDER BY timestamp
            """).df()
        one["timestamp"] = pd.to_datetime(one["timestamp"], utc=True).dt.tz_localize(None)
        one = one.set_index("timestamp").sort_index()
        # Derive cell voltage from pack voltage + cell series count
        one["cell_voltage_v"] = one["voltage_v"] / cells_lookup[sid]

        chem = str(chem_lookup.get(sid, "LFP"))
        sys_event_frames: list[pd.DataFrame] = []
        for rule_id, severity, channel, condition_fn, min_dur, sys_filter in RULES:
            if sys_filter is not None and sid not in sys_filter:
                continue
            allowed_chem = _RULE_CHEMISTRIES.get(rule_id)
            if allowed_chem is not None and chem not in allowed_chem:
                continue
            if channel not in one.columns:
                continue
            flag = condition_fn(one[channel])
            ev = collapse_to_events(flag, one[channel], min_duration_min=min_dur)
            if ev.empty:
                continue
            ev["rule_id"] = rule_id
            ev["severity"] = severity
            ev["channel"] = channel
            ev["system_id"] = sid
            sys_event_frames.append(ev)

        # Inverter-relative C-rate ceiling — a fraction-of-nameplate rule,
        # so it reads the same on a 2 kWh residential unit and a 1 MWh
        # system instead of misfiring on small high-C packs.
        ceiling = ceiling_lookup.get(sid)
        if ceiling is not None and "c_rate" in one.columns:
            flag = one["c_rate"] > ceiling * C_RATE_MARGIN
            ev = collapse_to_events(flag, one["c_rate"], min_duration_min=1)
            if not ev.empty:
                ev["rule_id"] = "c_rate_above_inverter"
                ev["severity"] = "warning"
                ev["channel"] = "c_rate"
                ev["system_id"] = sid
                sys_event_frames.append(ev)

        if sys_event_frames:
            n = sum(len(e) for e in sys_event_frames)
            print(f"  [{sid}] {n:,} events", flush=True)
            all_events.extend(sys_event_frames)
        else:
            print(f"  [{sid}] no events", flush=True)

    if not all_events:
        print("\nno events across the fleet — every rule dormant")
        return

    df = pd.concat(all_events, ignore_index=True)
    df = df[[
        "system_id", "rule_id", "severity", "channel",
        "start", "end", "duration_min", "peak_value",
    ]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_to_parquet(df, OUT_PATH, index=False, compression="snappy")
    print(f"\nwrote {OUT_PATH}: {len(df):,} events")

    print("\nSummary — rule × system event count:")
    summary = df.groupby(["rule_id", "system_id"]).size().unstack(fill_value=0)
    print(summary.to_string())


if __name__ == "__main__":
    main()
