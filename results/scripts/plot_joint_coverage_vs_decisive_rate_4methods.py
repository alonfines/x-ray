"""
Plot joint coverage vs. per-label decisive rate across the cluster-granularity
ladder (Rungs A-E: K=13, 12, 7, 6, 1), for four methods: max-rank, Bonferroni,
Sidak, LP-SCP (penalised arm, mu=nu=1, guard on).

This is the mirror of plot_joint_coverage_vs_ambiguity_4methods.py with the
x-axis un-inverted (decisive rate instead of ambiguity rate = 1-decisive).

Source: experiments/052_cluster_granularity_curve_3method/metrics.json,
rungs_summary_3method.csv (max-rank cited from experiments/051, not rerun).

Caveats carried into the plot as a footnote (do not drop when editing):
- B-D joint coverage has no formal guarantee (within-cluster-only
  calibration); only Rung A (independent) and Rung E (full joint) target
  0.90 by construction.
- LP-SCP numbers are not on the same footing as the other three: different
  nonconformity score (its own Mahalanobis measure, not the locked score)
  plus a structural Hp/Cp boost (mu=nu=1) the other methods do not get.
- Rung E's max-rank and LP-SCP joint coverage both fall short of the 0.90
  target on the 13-clean test population (0.8934, 0.8943); this shortfall
  is escalated to Tom in experiments/052's ESCALATION.md and not yet ruled
  on. Bonferroni/Sidak exceed target at Rung E (0.9154, 0.9111).

Follows CLAUDE.md's plot design guideline (large figure, bold oversized
fonts, high-contrast palette, value annotations, gridlines, 300 DPI).

Usage:
    .venv_exp048/bin/python results/scripts/plot_joint_coverage_vs_decisive_rate_4methods.py
"""

import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

RUNGS = ["A (K=13)", "B (K=12)", "C (K=7)", "D (K=6)", "E (K=1)"]

METHODS = {
    "max-rank": {
        "color": "#6EC6F0",
        "marker": "o",
        "dec": [0.6787, 0.6745, 0.5243, 0.5019, 0.2853],
        "cov": [0.4530, 0.4561, 0.5556, 0.5866, 0.8934],
    },
    "Bonferroni": {
        "color": "#F5A95B",
        "marker": "s",
        "dec": [0.6787, 0.6569, 0.4895, 0.4661, 0.2624],
        "cov": [0.4530, 0.4683, 0.5837, 0.6199, 0.9154],
    },
    "Sidak": {
        "color": "#C76EC6",
        "marker": "D",
        "dec": [0.6787, 0.6573, 0.4925, 0.4698, 0.2669],
        "cov": [0.4530, 0.4681, 0.5818, 0.6170, 0.9111],
    },
    "LP-SCP (mu=nu=1)": {
        "color": "#8CC63F",
        "marker": "^",
        "dec": [0.8044, 0.7984, 0.5649, 0.5422, 0.0252],
        "cov": [0.4456, 0.4514, 0.5517, 0.5837, 0.8943],
    },
}

TARGET = 0.90

fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor("white")

for name, m in METHODS.items():
    ax.plot(m["dec"], m["cov"], color=m["color"], marker=m["marker"],
             markersize=16, linewidth=3.2, label=name, zorder=3)
    for x, y, rung in zip(m["dec"], m["cov"], RUNGS):
        ax.annotate(rung.split(" ")[0], (x, y), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=14, fontweight="bold",
                    color=m["color"])

ax.axhline(TARGET, color="#555555", linestyle="--", linewidth=2.5, zorder=2)
ax.text(0.99, TARGET + 0.012, "90% nominal target", fontsize=18, fontweight="bold",
        color="#555555", ha="right", transform=ax.get_yaxis_transform())

ax.set_xlabel("Per-label decisive rate  (13×N denominator)", fontsize=22, fontweight="bold")
ax.set_ylabel("Joint coverage  (all 13 labels covered)", fontsize=22, fontweight="bold")
ax.set_title("Joint coverage vs. decisive rate across cluster granularity\n"
             "(Rungs A->E, test split, α=0.10)", fontsize=28, fontweight="bold", pad=18)

ax.invert_xaxis()
ax.set_xlim(1.02, -0.02)
ax.set_ylim(0.40, 1.0)
ax.yaxis.grid(True, linewidth=1.0, alpha=0.4, zorder=0)
ax.xaxis.grid(True, linewidth=1.0, alpha=0.2, zorder=0)
ax.tick_params(axis="both", labelsize=18)
ax.set_facecolor("white")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

legend = ax.legend(fontsize=18, loc="lower left", frameon=True)
for text in legend.get_texts():
    text.set_fontweight("bold")

footnote = (
    "B-D joint coverage is empirical only (no formal guarantee); only Rung A (independent) and Rung E (full joint) target 0.90 by construction.\n"
    "LP-SCP uses its own Mahalanobis score (not the locked score) plus a structural mu=nu=1 Hp/Cp boost the other methods don't get — not a clean 4-way score comparison.\n"
    "Rung E: max-rank (0.8934) and LP-SCP (0.8943) fall short of 0.90 on the 13-clean test population — escalated to Tom (exp052/ESCALATION.md), not yet ruled on."
)
fig.text(0.5, -0.02, footnote, fontsize=13, ha="center", va="top", color="#333333", wrap=True)

fig.tight_layout(rect=[0, 0.05, 1, 1])

out_path = "results/images/joint_coverage_vs_decisive_rate_4methods.png"
fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {out_path}")
