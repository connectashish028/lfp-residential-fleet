"""Degradation modes — cross-chemistry diagnostic view.

Surfaces the ICA/DVA degradation-mode pipeline: per-system capacity
observability, fade rate, and the LLI/LAM split, across every chemistry
in the lake (LFP / NMC / LMO-NMC). The headline is the observability
gate — the framework reports *whether* the field data can support a
degradation read before it reports the read itself.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from _components import charts, data, kpis, theme

st.set_page_config(page_title="Degradation · BESS Fleet Health",
                   layout="wide", initial_sidebar_state="expanded")
theme.inject(st)

kpis.hero_bar(
    brand="Degradation · cross-chemistry",
    badge="ICA / DVA · LLI vs LAM · observability gate",
)
st.markdown("<h1>Degradation-mode estimation</h1>", unsafe_allow_html=True)
st.markdown(
    "Quasi-OCV curves are reconstructed from low-dynamic field operation, "
    "differentiated into incremental-capacity (ICA) and differential-voltage "
    "(DVA) signatures, and capacity loss is attributed to **loss of lithium "
    "inventory (LLI)** vs **loss of active material (LAM)**. A mode is "
    "published **only when the capacity trend clears a confidence gate** "
    "(coefficient of variation ≤ 0.15 *and* fade R² ≥ 0.30) — so the view "
    "below is as much about *what can't be measured* as what can.",
    unsafe_allow_html=True,
)

summary = data.compute_degradation_summary()
if summary.empty:
    st.info(
        "No degradation-mode data found. Run "
        "`python -m bess_fleet.pipeline.degradation_modes` to build "
        "`data/curated/degradation_modes.parquet`."
    )
    st.stop()

modes = data.get_degradation_modes()

# ── Headline scatter ──────────────────────────────────────────────────
st.markdown("### Capacity observability splits by usage, not just chemistry")
st.plotly_chart(
    charts.degradation_observability(summary, height=440),
    width="stretch", config=charts.PLOTLY_CONFIG,
)
n_obs = int(summary["cap_observable"].sum())
st.caption(
    f"{n_obs} of {len(summary)} systems clear the gate (filled markers). "
    "**Red = LFP · blue = NMC · green = LMO/NMC.** Every LFP rack sits in the "
    "noisy upper band (several at physically-impossible negative fade); the "
    "systems that pass are NMC-family and land on the 2–3 %/yr literature band "
    "— but one NMC system fails too, because observability needs clean "
    "near-full cycles, not just a favourable chemistry."
)

# ── Per-system summary table ──────────────────────────────────────────
st.markdown("### Per-system verdict")
show = summary.assign(
    observable=summary["cap_observable"].map({True: "✅ yes", False: "— no"}),
).rename(columns={
    "system_id": "system", "n_months": "months", "peak_richness": "ICA peaks",
    "cap_cov": "capacity CoV", "fade_pct_per_yr": "fade %/yr",
    "fade_r2": "fade R²", "dominant_mode": "mode (heuristic)",
})[[
    "system", "chemistry", "months", "capacity CoV", "fade %/yr", "fade R²",
    "ICA peaks", "observable", "mode (heuristic)",
]]
st.dataframe(show, width="stretch", hide_index=True)

# ── Per-system capacity trend ─────────────────────────────────────────
st.markdown("### Capacity trend")
systems = summary["system_id"].tolist()
obs_first = (
    summary.sort_values("cap_observable", ascending=False)["system_id"].tolist()
)
pick = st.selectbox(
    "System", options=obs_first,
    format_func=lambda s: (
        f"{s} · {summary.loc[summary['system_id'] == s, 'chemistry'].iloc[0]}"
        + (" · observable" if bool(
            summary.loc[summary['system_id'] == s, 'cap_observable'].iloc[0]
        ) else " · low-confidence")
    ),
)
row = summary[summary["system_id"] == pick].iloc[0]
modes_sys = modes[modes["system_id"] == pick]

if bool(row["cap_observable"]):
    verdict = (
        f"**{pick}** ({row['chemistry']}) is **observable** — fade "
        f"{row['fade_pct_per_yr']:.2f} %/yr (R²={row['fade_r2']:.2f}). "
        f"Mechanism **{row['dominant_mode']}** *(heuristic — wants lab EIS / "
        "reference cycling to confirm)*."
    )
else:
    cov = row["cap_cov"]
    cov_str = f"{cov:.3f}" if pd.notna(cov) else "n/a (too few months)"
    verdict = (
        f"**{pick}** ({row['chemistry']}) is **low-confidence** — capacity "
        f"CoV {cov_str}, so the trend below is shown for transparency but no "
        "degradation mode is claimed from it."
    )
st.markdown(verdict, unsafe_allow_html=True)
if not modes_sys.empty:
    st.plotly_chart(
        charts.capacity_fade_trend(modes_sys, height=300),
        width="stretch", config=charts.PLOTLY_CONFIG,
    )

st.caption(
    "Method mirrors Figgener et al. (Nat Energy 2024, 10.1038/s41560-024-01620-9) "
    "and the degradation-mode follow-up (arXiv 2411.08025). The LLI/LAM split is "
    "a transparent heuristic; the observability gate and fade rate are the robust "
    "outputs."
)
