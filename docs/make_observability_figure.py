"""Generate the headline figure: field-capacity observability by chemistry.

Reads ``data/curated/degradation_modes.parquet`` (the degradation-mode
output), reduces it to the per-system verdict via the same
``system_summary`` the pipeline uses, and renders a single scatter that
tells the whole cross-chemistry story:

* y-axis  — capacity coefficient of variation (the confidence metric),
            log-scaled; lower is cleaner.
* x-axis  — annual capacity fade [%/yr]; the 2–3 %/yr literature band is
            shaded.
* a system is *observable* (filled marker) only if it clears the gate
  (CoV ≤ 0.15 and fade R² ≥ 0.30); otherwise it is hollow.

The reader sees at a glance that every LFP system sits in the noisy,
unobservable upper band (several at physically-impossible negative fade),
while the systems that clear the gate are NMC-family and land in the
literature fade range — except one NMC system whose gap-filled data keeps
it out, the point being that observability is *measured, not assumed*.

This is a dev/docs tool (matplotlib, not a runtime dependency). Run::

    python docs/make_observability_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_fleet.pipeline.degradation_modes import (  # noqa: E402
    CAP_COV_MAX,
    system_summary,
)

OUT_PNG = ROOT / "docs" / "degradation_observability.png"
COLORS = {"LFP": "#c0392b", "NMC": "#2471a3", "LMO": "#1e8449"}
CHEM_LABEL = {"LFP": "LFP", "NMC": "NMC", "LMO": "LMO/NMC"}


def main() -> None:
    modes = pd.read_parquet(ROOT / "data" / "curated" / "degradation_modes.parquet")
    rows = [
        system_summary(str(sid), str(grp["chemistry"].iloc[0]), grp)
        for sid, grp in modes.groupby("system_id")
    ]
    summ = pd.DataFrame(rows)
    plot = summ.dropna(subset=["cap_cov", "fade_pct_per_yr"]).copy()

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    # Literature fade band (2–3 %/yr) and the observability gate.
    ax.axvspan(2.0, 3.0, color="#f1c40f", alpha=0.12, zorder=0)
    ax.axhline(CAP_COV_MAX, ls="--", lw=1.2, color="#555", zorder=1)
    ax.axhspan(0.0, CAP_COV_MAX, color="#2ecc71", alpha=0.06, zorder=0)
    ax.text(
        -9.3, CAP_COV_MAX * 1.06, f"observability gate · CoV ≤ {CAP_COV_MAX}",
        fontsize=8.5, color="#555", va="bottom",
    )
    ax.text(2.5, 0.0105, "2–3 %/yr\nliterature", fontsize=8, color="#9a7d0a",
            ha="center", va="bottom")

    for chem, grp in plot.groupby("chemistry"):
        color = COLORS.get(chem, "#666")
        obs = grp[grp["cap_observable"]]
        non = grp[~grp["cap_observable"]]
        ax.scatter(non["fade_pct_per_yr"], non["cap_cov"], s=95,
                   facecolors="none", edgecolors=color, linewidths=1.6, zorder=3)
        ax.scatter(obs["fade_pct_per_yr"], obs["cap_cov"], s=110,
                   facecolors=color, edgecolors="k", linewidths=0.8, zorder=4,
                   label=f"{CHEM_LABEL.get(chem, chem)}")

    # Annotate every plotted system.
    for _, r in plot.iterrows():
        ax.annotate(
            r["system_id"],
            (r["fade_pct_per_yr"], r["cap_cov"]),
            textcoords="offset points", xytext=(7, 3), fontsize=8,
            color="#222",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Annual capacity fade  [%/yr]   (negative = physically impossible)")
    ax.set_ylabel("Capacity CoV  (lower = cleaner)   ·  log scale")
    ax.set_title(
        "Field-capacity observability splits by usage, not just chemistry",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-12, 19)
    # Build a legend with filled = observable, hollow = not. Park it in the
    # empty mid-left band so it never collides with a data point.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="chemistry (filled = observable)",
              loc="center left", bbox_to_anchor=(0.0, 0.42),
              fontsize=9, title_fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
