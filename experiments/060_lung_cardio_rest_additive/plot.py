"""Joint coverage and mean joint set size, from summary.csv.
Run: .venv/bin/python experiments/060_lung_cardio_rest_additive/plot.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
s = pd.read_csv(os.path.join(HERE, "summary.csv")).set_index("rule")
RULES = ["indep_bonferroni", "indep_max_rank", "lpscp_sidak", "mlp_sidak", "additive"]
LABEL = {"indep_bonferroni": "Bonferroni", "indep_max_rank": "Max-rank", "lpscp_sidak": "LP-SCP, 96.5% per cluster", "mlp_sidak": "MLP, 96.5% per cluster", "additive": "MLP, additive sum"}
COLORS = ["#CC79A7", "#882255", "#009E73", "#0072B2", "#E69F00"]   # Bonferroni pink, max-rank wine, MLP blue, additive orange

plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 13), facecolor="white", gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(len(RULES))


def bars(ax, col, scale, fmt, text_at):
    mean = s.loc[RULES, f"{col}_mean"].to_numpy() * scale
    lo, hi = s.loc[RULES, f"{col}_lo"].to_numpy() * scale, s.loc[RULES, f"{col}_hi"].to_numpy() * scale
    ax.bar(x, mean, 0.6, color=COLORS, yerr=[mean - lo, hi - mean], capsize=10, error_kw={"lw": 2.5})
    for xi, v, h in zip(x, mean, hi):
        ax.text(xi, text_at(h), fmt.format(v), ha="center", va="bottom", fontsize=22, fontweight="bold")
    ax.set_xticks(x, [LABEL[r] for r in RULES], rotation=45, ha="right", fontsize=20)
    ax.tick_params(axis="y", labelsize=19)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# Coverage panel: per-cluster bars (lung, cardio, rest) + joint bar. The additive sum sets one
# threshold on the summed score, so it has no per-cluster target: joint bar only.
GROUPS = [("coverage_lung", "Lung"), ("coverage_cardio", "Cardio"), ("coverage_rest", "Rest"),
          ("joint_coverage", "Joint (13)")]
GPAL = ["#0072B2", "#D55E00", "#009E73", "#7B3294"]     # lung, cardio, rest, joint (Okabe-Ito)
w = 0.2
for g, (col, name) in enumerate(GROUPS):
    rules = RULES if col == "joint_coverage" else RULES[:-1]
    xs = np.array([RULES.index(r) for r in rules], float)
    xs = xs + (g - 1.5) * w if col != "joint_coverage" else xs + np.where(
        np.array(rules) == "additive", 0, 1.5 * w)
    vals = s.loc[rules, f"{col}_mean"].to_numpy() * 100
    ax1.bar(xs, vals, w, color=GPAL[g], label=name,
            edgecolor="black" if col == "joint_coverage" else "none", linewidth=2.5)
    for xi, v in zip(xs, vals):
        ax1.text(xi, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=17, fontweight="bold",
                 rotation=90 if col != "joint_coverage" else 0)
ax1.axhline(96.5, color="dimgray", ls=":", lw=3.5, label="96.5% per cluster")
ax1.axhline(90, color="crimson", ls="--", lw=3, label="90% joint target")
ax1.set_xticks(x, [LABEL[r] for r in RULES], rotation=45, ha="right", fontsize=20)
ax1.set_ylim(85, 104)
ax1.set_ylabel("Coverage (%)", fontsize=24)
ax1.set_title("Coverage: each cluster at 96.5%, joint at 90%", fontsize=28, pad=18)
ax1.legend(fontsize=17, ncol=3, loc="upper left", frameon=False)
ax1.tick_params(axis="y", labelsize=19)
ax1.yaxis.grid(True, alpha=0.4)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)

bars(ax2, "mean_size", 1, "{:,.0f}", lambda h: h * 1.15)
ax2.set_yscale("log")
ax2.set_ylim(10, 20000)
ax2.set_ylabel("Mean joint set size (log scale)", fontsize=24)
ax2.set_title("Mean joint set size", fontsize=30, pad=18)

fig.suptitle("Lung / cardio / rest clusters, 90% joint coverage", fontsize=32, fontweight="bold", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=8)
fig.savefig(os.path.join(HERE, "coverage_and_size.png"), dpi=300, facecolor="white")
