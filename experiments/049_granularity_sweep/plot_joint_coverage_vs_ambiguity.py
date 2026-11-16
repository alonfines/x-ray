#!/usr/bin/env python3
"""
Plot joint coverage vs. ambiguity rate (1 - fully_decisive_rate) for Exp049 headline methods.
Adds Exp050's direct result as a separate point.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

EXPERIMENT_DIR = os.path.dirname(__file__)
RESULTS_RAW_PATH = os.path.join(EXPERIMENT_DIR, "results_raw.csv")
EXP050_RESULTS_PATH = os.path.join(os.path.dirname(EXPERIMENT_DIR), "050_direct_joint_coverage", "results.json")

# Plot style
rcParams['font.size'] = 18
rcParams['axes.labelsize'] = 22
rcParams['axes.titlesize'] = 28
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 17
rcParams['figure.facecolor'] = 'white'

def load_results():
    """Load results_raw.csv and compute label-weighted aggregates."""
    df = pd.read_csv(RESULTS_RAW_PATH)

    # Keep only headline methods and their primary variants
    # max_rank: no variant
    # per_label: bonferroni variant
    # lp_scp: admissible variant
    headline_data = []

    for level in sorted(df['level'].unique()):
        level_df = df[df['level'] == level]
        n_clusters = level_df['n_clusters'].iloc[0]

        for method in ['max_rank', 'per_label', 'lp_scp']:
            if method == 'max_rank':
                # max_rank has no variant column
                method_df = level_df[
                    (level_df['method'] == 'max_rank') &
                    (level_df['variant'].isna() | (level_df['variant'] == ''))
                ]
            elif method == 'per_label':
                method_df = level_df[
                    (level_df['method'] == 'per_label') &
                    (level_df['variant'] == 'bonferroni')
                ]
            elif method == 'lp_scp':
                method_df = level_df[
                    (level_df['method'] == 'lp_scp') &
                    (level_df['variant'] == 'admissible')
                ]

            if len(method_df) == 0:
                continue

            # Compute label-weighted aggregate
            # Weight each row by its m (cluster size)
            m_values = method_df['m'].values
            total_m = m_values.sum()

            joint_cov = (method_df['joint_coverage'].values * m_values).sum() / total_m
            fully_decisive = (method_df['fully_decisive_rate'].fillna(0).values * m_values).sum() / total_m

            headline_data.append({
                'level': level,
                'n_clusters': n_clusters,
                'method': method,
                'joint_coverage': joint_cov,
                'fully_decisive_rate': fully_decisive,
            })

    return pd.DataFrame(headline_data)

def load_exp050():
    """Load Exp050's direct joint coverage result."""
    with open(EXP050_RESULTS_PATH) as f:
        data = json.load(f)
    return data['direct_joint_coverage'], data['fully_decisive_rate']

def main():
    # Load data
    headline_df = load_results()
    exp050_cov, exp050_decisive = load_exp050()
    exp050_ambiguity = 1.0 - exp050_decisive

    # Compute ambiguity rate
    headline_df['ambiguity_rate'] = 1.0 - headline_df['fully_decisive_rate']

    # Create plot
    fig, ax = plt.subplots(figsize=(18, 11), dpi=100)

    # Define colors for methods (mako palette inspired)
    colors = {
        'max_rank': '#3B4CC0',     # Blue
        'per_label': '#F7F056',    # Yellow
        'lp_scp': '#B35806',       # Orange
    }

    markers = {
        'max_rank': 'o',
        'per_label': 's',
        'lp_scp': '^',
    }

    # Plot each method as a line
    for method in ['max_rank', 'per_label', 'lp_scp']:
        method_data = headline_df[headline_df['method'] == method].sort_values('level')
        ax.plot(
            method_data['ambiguity_rate'],
            method_data['joint_coverage'],
            marker=markers[method],
            markersize=10,
            linewidth=2.5,
            color=colors[method],
            label=method.replace('_', ' ').title(),
        )

    # Add Exp050's direct point (distinctly marked)
    ax.scatter(
        [exp050_ambiguity],
        [exp050_cov],
        marker='*',
        s=800,
        color='#E41A1C',  # Red
        edgecolors='black',
        linewidth=2,
        label='Direct, no correction, m=1×13',
        zorder=10,
    )

    # Add reference line at 0.90 coverage
    ax.axhline(y=0.90, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.90)')

    # Labels and formatting
    ax.set_xlabel('Ambiguity Rate (1 − Fully Decisive Rate)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Joint Coverage', fontsize=22, fontweight='bold')
    ax.set_title('Joint Coverage vs. Ambiguity Rate Across Clustering Granularities', fontsize=28, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    # Y-axis limits and ticks
    ax.set_ylim([0.40, 0.95])
    ax.set_yticks(np.arange(0.40, 1.0, 0.05))

    # X-axis limits
    ax.set_xlim([-0.05, 1.05])

    # Legend
    ax.legend(loc='best', framealpha=0.95, fontsize=17, edgecolor='black', fancybox=False)

    # Tight layout
    plt.tight_layout()

    # Save at high DPI
    output_path = os.path.join(EXPERIMENT_DIR, 'joint_coverage_vs_ambiguity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_path}")

    return output_path

if __name__ == '__main__':
    main()
