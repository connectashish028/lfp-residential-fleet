"""Capacity / SOHc — Figgener field-capacity method (eq 1–3).

Surfaces the absolute usable-capacity estimator: per-system SOHc
estimates over time with the ageing trend and a 75 % CI band (the paper's
Fig 4), a cross-system verdict table, and a reliability gate. Distinct
from the Degradation page (ICA/DVA mechanism); this is the *how much*
capacity is left, validated against the paper's ageing rates.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from _components import charts, data, kpis, theme

st.set_page_config(page_title="SOHc · BESS Fleet Health",
                   layout="wide", initial_sidebar_state="expanded")
theme.inject(st)

kpis.hero_bar(
    brand="Capacity / SOHc · Figgener method",
    badge="ECM OCV · offset-corrected coulomb counting · ageing rate ± CI",
)
st.markdown("<h1>Usable-capacity / SOHc estimation</h1>", unsafe_allow_html=True)
st.markdown(
    "A field replication of Figgener et al. (*Nat Energy* 2024, eq 1–3): "
    "relaxation OCV from a 2nd-order ECM → offset-corrected coulomb counting "
    "between full (EOC) and empty (EOD) → usable capacity → **SOHc** → an "
    "inverse-variance-weighted ageing rate. Each estimate carries a σ "
    "propagated from the OCV-fit error; a system's rate is published only "
    "behind a **reliability gate** (enough clean full cycles, low σ, tight CI).",
    unsafe_allow_html=True,
)

soh = data.get_capacity_soh()
est = data.get_capacity_estimates()
if soh.empty:
    st.info(
        "No capacity estimates found. Run "
        "`python -m bess_fleet.pipeline.capacity_estimation`."
    )
    st.stop()

# ── Reliable ageing rates vs the paper ────────────────────────────────
st.markdown("### Reliable ageing rates — validated against the paper")
n_rel = int(soh["reliable"].sum())
st.caption(
    f"{n_rel} of {len(soh)} systems clear the reliability gate. The paper "
    "reports SLMO 2.1–3.1, MNMC 1.9–3.2, MLFP 2.0–2.2 pp/yr; the reliable "
    "systems here land in/near that band with no calibration. The σ flags "
    "*per-system* noise (e.g. ID01 fails despite being LMO like the reliable "
    "ID02) — more useful than a chemistry-level CI."
)

# ── Per-system verdict table ──────────────────────────────────────────
show = soh.assign(
    reliable_disp=soh["reliable"].map({True: "✅ yes", False: "— no"}),
    rate=soh.apply(
        lambda r: (
            f"{r['ageing_pct_per_yr']:.2f} ± {r['ageing_ci95_pp_yr']:.2f}"
            if pd.notna(r["ageing_pct_per_yr"]) else "—"
        ),
        axis=1,
    ),
).rename(columns={
    "system_id": "system", "n_estimates": "estimates",
    "mean_sigma_pp": "mean σ (pp)", "soh_latest_pct": "SOHc latest %",
})[[
    "system", "chemistry", "estimates", "rate", "mean σ (pp)",
    "SOHc latest %", "reliable_disp",
]].rename(columns={"rate": "ageing pp/yr (±95%CI)", "reliable_disp": "reliable"})
st.dataframe(show, width="stretch", hide_index=True)

# ── Per-system SOHc time series (paper Fig 4) ─────────────────────────
st.markdown("### Capacity fade over time")
have_est = sorted(est["system_id"].unique()) if not est.empty else []
order = (
    soh.sort_values("reliable", ascending=False)["system_id"].tolist()
)
order = [s for s in order if s in have_est]
if not order:
    st.info("No per-estimate series to plot.")
    st.stop()
pick = st.selectbox(
    "System", options=order,
    format_func=lambda s: (
        f"{s} · {soh.loc[soh['system_id'] == s, 'chemistry'].iloc[0]}"
        + (" · reliable" if bool(
            soh.loc[soh['system_id'] == s, 'reliable'].iloc[0]
        ) else " · low-confidence")
    ),
)
row = soh[soh["system_id"] == pick].iloc[0]
est_sys = est[est["system_id"] == pick]

verdict = (
    f"**{pick}** ({row['chemistry']}) — {int(row['n_estimates'])} estimates, "
    f"ageing **{row['ageing_pct_per_yr']:.2f} ± {row['ageing_ci95_pp_yr']:.2f} pp/yr**, "
    f"mean σ {row['mean_sigma_pp']:.1f} pp. "
    + (
        "**Reliable.**" if bool(row["reliable"])
        else "**Low-confidence** — shown for transparency; the rate doesn't "
        "clear the reliability gate (too few clean full cycles / σ too high)."
    )
)
st.markdown(verdict, unsafe_allow_html=True)
if not est_sys.empty:
    st.plotly_chart(
        charts.capacity_soh_timeseries(est_sys, float(row["ageing_pct_per_yr"]), height=360),
        width="stretch", config=charts.PLOTLY_CONFIG,
    )

st.caption(
    "Method note: the ECM OCV is fit at 1-minute cadence — validated to match "
    "the faithful 1-second 2nd-order fit to ~0.1 mV (the fast RC decays before "
    "the first 1-min sample), so the fleet runs with no raw-data I/O. Figgener "
    "et al., Nat Energy 2024 (10.1038/s41560-024-01620-9)."
)
