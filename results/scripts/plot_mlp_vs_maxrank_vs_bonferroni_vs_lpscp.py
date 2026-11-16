"""
Joint coverage and mean joint set size: Bonferroni vs max-rank vs LP-SCP vs the
learned MLP joint-state score (L1), on cardio_pair, pleural_free and lung_wide.

Source: experiments/055_learned_score_full/summary_by_arm.csv — all four methods on
the SAME rows and the SAME 100 subject-level 50/50 resplits of pooled all-13-clean
conformal + test (Exp053 protocol). Bars = resplit mean; error bars = 2.5/97.5
resplit percentiles. MLP = arm L1, seed 0 (the pre-registered primary seed).
LP-SCP = admissible arm, mu = nu = 1 (pre-registered primary; NOT the mu=nu=2 grid
cell of maxrank_vs_bonferroni_vs_lpscp_mu2nu2.png). max-rank = plain product set (no
admissibility pruning); the max-rank ∩ admissible numbers are printed for reference.

Scope: all-13-clean studies (77% of test). Shared-backbone caveat applies: the
efficiency gains likely overstate what independently trained models would give.

Follows CLAUDE.md's plot design guideline (large figure, bold oversized fonts,
value annotations, gridlines, white background, 300 DPI). Colours validated with the
dataviz palette validator (light mode, all checks pass; Bonferroni orange has
<3:1 contrast, relieved by value labels and tick names).

Usage:
    /usr/bin/python3 results/scripts/plot_mlp_vs_maxrank_vs_bonferroni_vs_lpscp.py
"""

import os
import textwrap

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "experiments/055_learned_score_full/summary_by_arm.csv")
OUT = os.path.join(ROOT, "results/images/mlp_vs_maxrank_vs_bonferroni_vs_lpscp.png")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

CLUSTERS = [
    ("cardio_pair", "Enlarged Cardiomediastinum, Cardiomegaly"),
    ("pleural_free", "Pneumothorax, Pleural Effusion, Support Devices"),
    ("lung_wide", "Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis"),
]
# (variant in CSV, tick label, colour) — colour follows the method across panels
METHODS = [
    ("bonferroni", "Bonferroni", "#D9822B"),
    ("maxrank", "max-rank", "#2B8CC4"),
    ("lpscp_adm", "LP-SCP\n(µ=ν=1)", "#5E9E1F"),
    ("L1", "MLP\n(learned)", "#6A3FA0"),
]
TARGET = 0.90
INK = "#222222"
MUTED = "#555555"

df = pd.read_csv(SRC)
df = df[df.family == "designed"]


def row(unit, variant):
    r = df[(df.unit == unit) & (df.variant == variant)]
    if len(r) > 1:  # learned arms have seeds 0-4; reference arms have seed -1
        r = r[r.seed == 0]
    assert len(r) == 1, (unit, variant, len(r))
    return r.iloc[0]


fig, axes = plt.subplots(2, len(CLUSTERS), figsize=(25, 13))
fig.patch.set_facecolor("white")
x = list(range(len(METHODS)))
ticks = [m[1] for m in METHODS]
colors = [m[2] for m in METHODS]

for col, (unit, labels) in enumerate(CLUSTERS):
    rows = [row(unit, v) for v, _, _ in METHODS]
    m = int(rows[0].m)
    n_states = 2 ** m

    for r_i, (metric, ylab, fmt) in enumerate([
        ("joint_coverage", "Joint coverage", "{:.3f}"),
        ("mean_joint_set_size", "Mean joint set size", "{:.2f}"),
    ]):
        ax = axes[r_i, col]
        mean = [r[f"{metric}_mean"] for r in rows]
        lo = [r[f"{metric}_mean"] - r[f"{metric}_p2.5"] for r in rows]
        hi = [r[f"{metric}_p97.5"] - r[f"{metric}_mean"] for r in rows]
        ax.bar(x, mean, width=0.62, color=colors, zorder=3, edgecolor="white", linewidth=2)
        ax.errorbar(x, mean, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=2.5,
                    capsize=8, capthick=2.5, zorder=4)
        for xi, v, h in zip(x, mean, hi):
            ax.annotate(fmt.format(v), (xi, v + h), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=18, fontweight="bold", color=INK)

        if metric == "joint_coverage":
            ax.axhline(TARGET, color=MUTED, linestyle="--", linewidth=2.5, zorder=2)
            if col == len(CLUSTERS) - 1:  # label once, outside the plot, clear of the bars
                ax.text(1.02, TARGET, "target\n0.90", fontsize=17, fontweight="bold",
                        color=MUTED, ha="left", va="center", transform=ax.get_yaxis_transform())
            ax.set_ylim(0.80, 1.0)
            ax.set_title(f"{unit}  (m={m})", fontsize=30, fontweight="bold", pad=16)
        else:
            ax.set_ylim(0, max(mean) * 1.25)
            wrapped = "\n".join(textwrap.wrap(labels, width=30))
            ax.set_xlabel(f"out of {n_states} joint states\n{wrapped}",
                          fontsize=17, fontweight="bold", labelpad=12)

        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=18, fontweight="bold")
        if col == 0:
            ax.set_ylabel(ylab, fontsize=24, fontweight="bold")
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.grid(True, linewidth=1.0, alpha=0.4, zorder=0)
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    adm = row(unit, "maxrank_adm")
    print(f"{unit}: max-rank ∩ admissible size {adm.mean_joint_set_size_mean:.3f}, "
          f"coverage {adm.joint_coverage_mean:.4f}")

fig.suptitle("Bonferroni vs Max-Rank vs LP-SCP vs MLP", fontsize=30, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(hspace=0.32, wspace=0.22)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {OUT}")
