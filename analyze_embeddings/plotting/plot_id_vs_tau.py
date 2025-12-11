#!/usr/bin/env python
"""
Plotting Script: Aggregated Fidelity (Tau) vs. Granularity (ID).
Averages metrics across all progressions to show global model trends.
Includes Error Bars (Standard Error of the Mean).
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
pcfg.set_icml_style()

INPUT_FILE = config.FULL_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EMBEDDING = 'final_embedding'
X_METRIC = 'id_raw'
Y_METRIC = 'tau'

# Mapping Definitions
SHAPE_MAPPING = {
    'DINOv2 Baseline': 'D',
    'Vision-Language (Path)': 'X',
    'Vision-Only (Path)': 'o'
}
SHAPE_ORDER = ['Vision-Only (Path)', 'Vision-Language (Path)', 'DINOv2 Baseline']

def plot_aggregated_scatter():
    # 1. Load & Filter Data
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    df = df[df['embedding_type'] == TARGET_EMBEDDING].copy()
    df = df[df['model'].isin(pcfg.MODEL_ORDER)]
    
    # 2. Add Meta-Data
    df['model_type'] = df['model'].map(pcfg.MODEL_TYPE_MAPPING)

    # 3. Aggregate Data (Mean + SEM)
    # Group by Model to get the average performance across all diseases
    agg_df = df.groupby(['model', 'model_type'], observed=True)[[X_METRIC, Y_METRIC]].agg(
        mean_x=(X_METRIC, 'mean'),
        sem_x=(X_METRIC, 'sem'),
        mean_y=(Y_METRIC, 'mean'),
        sem_y=(Y_METRIC, 'sem'),
        count=(X_METRIC, 'count')
    ).reset_index()

    # 4. Plot Setup
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 5. Plot Error Bars first (zorder=1 to put behind markers)
    # We loop manually to color the error bars correctly
    for _, row in agg_df.iterrows():
        color = pcfg.MODEL_COLORS.get(row['model'], 'black')
        
        ax.errorbar(
            x=row['mean_x'],
            y=row['mean_y'],
            xerr=row['sem_x'],
            yerr=row['sem_y'],
            fmt='none',      # No marker drawn by errorbar (we add it separately)
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            alpha=0.6,
            zorder=1
        )

    # 6. Plot Markers (zorder=2 to sit on top)
    sns.scatterplot(
        data=agg_df,
        x='mean_x',
        y='mean_y',
        hue='model',
        style='model_type',
        palette=pcfg.MODEL_COLORS,
        markers=SHAPE_MAPPING,
        hue_order=pcfg.MODEL_ORDER,
        style_order=SHAPE_ORDER,
        s=300,
        alpha=0.9,
        edgecolor='k',
        linewidth=0.8,
        ax=ax,
        legend=False,
        zorder=2
    )

    # 7. Formatting
    ax.set_title("Manifold Efficiency:\nAggregated Performance Across All Diseases", 
                 fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel("Average Representation Granularity (Intrinsic Dim)", fontweight='bold', labelpad=10)
    ax.set_ylabel(r"Average Trajectory Fidelity ($\tau$)", fontweight='bold', labelpad=10)
    
    # Reverse X-axis if you want "Compression" to go left-to-right? 
    # Usually strictly plotting values is safer.
    
    ax.grid(True, linestyle='--', alpha=0.3)
    sns.despine(ax=ax, trim=True)

    # 8. Legends (Stacked)
    # Shape Legend
    shape_handles = [
        Line2D([0], [0], marker=SHAPE_MAPPING[m], color='w', label=m,
               markerfacecolor='#777777', markeredgecolor='k', markersize=12)
        for m in SHAPE_ORDER
    ]
    legend_shape = fig.legend(
        handles=shape_handles, loc='upper center', bbox_to_anchor=(0.5, -0.01), 
        ncol=1, frameon=False, fontsize=11, title="Model Types"
    )

    # Color Legend
    color_handles = [
        Line2D([0], [0], marker='o', color='w', label=pcfg.MODEL_LABELS.get(m, m),
               markerfacecolor=pcfg.MODEL_COLORS.get(m, 'k'), markersize=10, markeredgecolor='k')
        for m in pcfg.MODEL_ORDER
    ]
    fig.legend(
        handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
        ncol=3, frameon=False, fontsize=11, title="Specific Models"
    )
    
    fig.add_artist(legend_shape)

    # Adjust layout to make room for bottom legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    save_path = OUTPUT_DIR / 'aggregated_id_vs_tau.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_aggregated_scatter()