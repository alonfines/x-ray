"""
Plot joint coverage and mean joint set size, max-rank vs Bonferroni vs LP-SCP,
on the three label clusters: cardio_pair, pleural_free, lung_wide (lung_core
omitted per request). Source numbers: experiments/046/REPORT.md
(max-rank/Bonferroni), experiments/047/lp_scp_results.csv,
arm=grid_mu2.0_nu2.0+guard (LP-SCP, mu=nu=2, exploratory grid cell).

NOTE: the mu=nu=2 grid rows in lp_scp_results.csv have enforce_hierarchy=
False — i.e. this is the penalised-style LP-SCP arm (soft Hp/Cp penalties
only), NOT the hard-constraint `admissible` arm used in the mu=nu=1 plot.
Per experiments/047/REPORT.md, (2,2) was the grid-minimizing cell but was
explicitly NOT adopted as primary (pre-registered mu=nu=1 stands); this plot
is the exploratory-appendix comparison, not a replacement headline.

Follows CLAUDE.md's plot design guideline (large figure, bold oversized
fonts, high-contrast palette, value annotations, gridlines, 300 DPI).

Usage:
    .venv-extraction/bin/python results/scripts/plot_maxrank_vs_bonferroni_vs_lpscp.py
"""

import textwrap

import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

CLUSTERS = [
    {
        "name": "cardio_pair",
        "labels": "Enlarged Cardiomediastinum, Cardiomegaly",
        "m": 2,
        "n": 23128,
        "max_cov": 0.8959, "bonf_cov": 0.9421, "lpscp_cov": 0.9012,
        "max_size": 2.260, "bonf_size": 2.623, "lpscp_size": 1.305,
        "n_states": 4,
    },
    {
        "name": "pleural_free",
        "labels": "Pneumothorax, Pleural Effusion, Support Devices",
        "m": 3,
        "n": 23630,
        "max_cov": 0.8982, "bonf_cov": 0.9036, "lpscp_cov": 0.8987,
        "max_size": 3.726, "bonf_size": 3.802, "lpscp_size": 3.053,
        "n_states": 8,
    },
    {
        "name": "lung_wide",
        "labels": "Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis",
        "m": 5,
        "n": 20318,
        "max_cov": 0.8993, "bonf_cov": 0.9242, "lpscp_cov": 0.8961,
        "max_size": 14.839, "bonf_size": 16.563, "lpscp_size": 7.432,
        "n_states": 32,
    },
]

COLOR_MAXRANK = "#6EC6F0"
COLOR_BONFERRONI = "#F5A95B"
COLOR_LPSCP = "#8CC63F"
COLOR_TARGET = "#555555"
TARGET_ALPHA = 0.10

fig, axes = plt.subplots(2, len(CLUSTERS), figsize=(20, 11))
fig.patch.set_facecolor("white")

bar_width = 0.6
x = [0, 1, 2]
xticklabels = ["max-rank", "Bonferroni", "LP-SCP\n(µ=ν=2)"]

for col, c in enumerate(CLUSTERS):
    # --- Row 1: joint coverage ---
    ax = axes[0, col]
    values = [c["max_cov"], c["bonf_cov"], c["lpscp_cov"]]
    colors = [COLOR_MAXRANK, COLOR_BONFERRONI, COLOR_LPSCP]
    bars = ax.bar(x, values, width=bar_width, color=colors, zorder=3)
    ax.axhline(1 - TARGET_ALPHA, color=COLOR_TARGET, linestyle="--", linewidth=2.5, zorder=2)
    ax.text(0.02, 1 - TARGET_ALPHA - 0.065, "target 0.90", fontsize=16,
            fontweight="bold", color=COLOR_TARGET, ha="left",
            transform=ax.get_yaxis_transform())

    for xi, v in zip(x, values):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=17, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=18, fontweight="bold")
    ax.set_title(f"{c['name']}  (m={c['m']})", fontsize=28, fontweight="bold", pad=14)
    if col == 0:
        ax.set_ylabel("Joint coverage", fontsize=22, fontweight="bold")
    ax.yaxis.grid(True, linewidth=1.0, alpha=0.4, zorder=0)
    ax.set_facecolor("white")
    ax.tick_params(axis="y", labelsize=18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # --- Row 2: mean joint set size ---
    ax2 = axes[1, col]
    values2 = [c["max_size"], c["bonf_size"], c["lpscp_size"]]
    bars2 = ax2.bar(x, values2, width=bar_width, color=colors, zorder=3)

    for xi, v in zip(x, values2):
        ax2.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=17, fontweight="bold")

    ax2.set_ylim(0, c["n_states"] * 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels, fontsize=18, fontweight="bold")
    if col == 0:
        ax2.set_ylabel("Mean joint set size", fontsize=22, fontweight="bold")
    wrapped_labels = "\n".join(textwrap.wrap(c["labels"], width=26))
    ax2.set_xlabel(f"out of {c['n_states']} joint states\n{wrapped_labels}",
                   fontsize=15, fontweight="bold", labelpad=10)
    ax2.yaxis.grid(True, linewidth=1.0, alpha=0.4, zorder=0)
    ax2.set_facecolor("white")
    ax2.tick_params(axis="y", labelsize=18)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

fig.suptitle("Max-rank vs Bonferroni vs LP-SCP (µ=ν=2, grid, not primary) — "
             "joint coverage and set size (test split, α=0.10)",
             fontsize=22, fontweight="bold", y=1.03)
fig.subplots_adjust(wspace=0.38, hspace=0.45)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

out_path = "results/images/maxrank_vs_bonferroni_vs_lpscp_mu2nu2.png"
fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {out_path}")
