"""
Plot per-label decisive rate vs. decisive accuracy (among fully-decisive
rows), across Rungs A-D, for four methods: max-rank, Bonferroni, Sidak,
LP-SCP (penalised arm, mu=nu=1, guard on).

Rung E is excluded from this plot: decisive accuracy there is undefined
(n=0 fully-decisive rows) for Bonferroni, Sidak and LP-SCP, and noisy
(n=2) for max-rank, per Jules's explicit "do not read" note.

Decisive accuracy is a POST-HOC addition requested by Alon, not one of
Tom's 5 locked metrics for experiments/052. Definition: among rows with
all 13 labels decisive (per-label projection rule), the fraction whose
decisive 13-label vector exactly equals the true 13-label vector.

Source: experiments/052_cluster_granularity_curve_3method/
decisive_acc_fully_decisive_addendum.csv (max-rank cited from
experiments/051, not rerun).

Caveats carried into the plot as a footnote (do not drop when editing):
- The rise in accuracy from A->D is largely a SELECTION EFFECT: the
  fully-decisive subset shrinks to the easiest rows at coarser rungs, not
  an improvement in the method itself (Jules's explicit note).
- n (fully-decisive row count) shrinks sharply across rungs (~1500 at A to
  a few hundred at D) and is annotated at every point; small-n points are
  noisier than the smooth-looking line suggests.
- LP-SCP numbers are not on the same footing as the other three: different
  nonconformity score plus a structural mu=nu=1 Hp/Cp boost the other
  methods do not get.

Follows CLAUDE.md's plot design guideline (large figure, bold oversized
fonts, high-contrast palette, value annotations, gridlines, 300 DPI).

Usage:
    .venv_exp048/bin/python results/scripts/plot_decisive_rate_vs_decisive_accuracy_4methods.py
"""

import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

RUNGS = ["A", "B", "C", "D"]

# dec = per-label decisive rate (13xN); acc = decisive accuracy among
# fully-decisive rows; n = fully-decisive row count (annotated, not Rung E)
METHODS = {
    "max-rank": {
        "color": "#6EC6F0",
        "marker": "o",
        "dec": [0.6787, 0.6745, 0.5243, 0.5019],
        "acc": [0.8252, 0.8339, 0.8783, 0.8803],
        "n":   [1510, 1481, 378, 376],
    },
    "Bonferroni": {
        "color": "#F5A95B",
        "marker": "s",
        "dec": [0.6787, 0.6569, 0.4895, 0.4661],
        "acc": [0.8252, 0.8560, 0.8992, 0.9028],
        "n":   [1510, 1368, 248, 247],
    },
    "Sidak": {
        "color": "#C76EC6",
        "marker": "D",
        "dec": [0.6787, 0.6573, 0.4925, 0.4698],
        "acc": [0.8252, 0.8562, 0.9015, 0.9049],
        "n":   [1510, 1370, 264, 263],
    },
    "LP-SCP (mu=nu=1)": {
        "color": "#8CC63F",
        "marker": "^",
        "dec": [0.8044, 0.7984, 0.5649, 0.5422],
        "acc": [0.7699, 0.7739, 0.8988, 0.8988],
        "n":   [5146, 5091, 761, 761],
    },
}

fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor("white")

for name, m in METHODS.items():
    ax.plot(m["dec"], m["acc"], color=m["color"], marker=m["marker"],
             markersize=16, linewidth=3.2, label=name, zorder=3)
    for x, y, rung, n in zip(m["dec"], m["acc"], RUNGS, m["n"]):
        ax.annotate(f"{rung} (n={n})", (x, y), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=13, fontweight="bold",
                    color=m["color"])

ax.set_xlabel("Per-label decisive rate  (13×N denominator)", fontsize=22, fontweight="bold")
ax.set_ylabel("Decisive accuracy  (among fully-decisive rows)", fontsize=22, fontweight="bold")
ax.set_title("Decisive rate vs. decisive accuracy, Rungs A-D\n"
             "(test split, α=0.10; Rung E excluded — undefined/noisy)",
             fontsize=28, fontweight="bold", pad=18)

ax.invert_xaxis()
ax.set_xlim(0.90, 0.40)
ax.set_ylim(0.70, 1.0)
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
    "Post-hoc metric, not one of the 5 locked metrics for exp052. The rise in accuracy from A->D is largely a SELECTION EFFECT — the fully-decisive\n"
    "subset shrinks to the easiest rows at coarser rungs, not an improvement in the method itself. n (fully-decisive rows) shrinks sharply, annotated per point.\n"
    "LP-SCP uses its own Mahalanobis score plus a structural mu=nu=1 Hp/Cp boost the other methods don't get — not a clean 4-way comparison."
)
fig.text(0.5, -0.02, footnote, fontsize=13, ha="center", va="top", color="#333333", wrap=True)

fig.tight_layout(rect=[0, 0.05, 1, 1])

out_path = "results/images/decisive_rate_vs_decisive_accuracy_4methods.png"
fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {out_path}")
