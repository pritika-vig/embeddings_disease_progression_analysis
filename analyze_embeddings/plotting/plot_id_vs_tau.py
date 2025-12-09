#!/usr/bin/env python
"""
Plotting Script: Fidelity (Tau) vs. Intrinsic Dimension (Granularity).
Tests Hypothesis: Does semantic compression (Low ID) hurt trajectory recovery?
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
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
X_METRIC = 'id_raw'  # Intrinsic Dimension of the raw representation
Y_METRIC = 'tau'     # Trajectory Fidelity

# --- Shape Mappings (Same as before) ---
SHAPE_MAPPING = {
    'DINOv2 Baseline': 'D',
    'Vision-Language (Path)': 'X',
    'Vision-Only (Path)': 'o'
}
SHAPE_ORDER = ['Vision-Only (Path)', 'Vision-Language (Path)', 'DINOv2 Baseline']

def plot_id_vs_fidelity():
    # 1. Load Data
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # Filter
    df = df[df['embedding_type'] == TARGET_EMBEDDING].copy()
    df = df[df['model'].isin(pcfg.MODEL_ORDER)]
    
    # Categorical Ordering
    df['model'] = pd.Categorical(df['model'], categories=pcfg.MODEL_ORDER, ordered=True)
    df['model_type'] = df['model'].map(pcfg.MODEL_TYPE_MAPPING)
    df = df.sort_values(['model_type', 'model'])

    # Targets
    target_progs = [p for p in pcfg.PROGRESSION_COLORS.keys() if p != 'Null']

    # 2. Setup Grid
    cols = 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 11), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    # 3. Plot Loop
    for i, prog in enumerate(target_progs):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        
        subset = df[df['progression'] == prog].copy()
        
        # Calc Correlation
        valid_data = subset.dropna(subset=[X_METRIC, Y_METRIC])
        corr_str = ""
        if len(valid_data) > 2:
            r, p_val = pearsonr(valid_data[X_METRIC], valid_data[Y_METRIC])
            corr_str = f"\n(Pearson r={r:.2f})"
        
        sns.scatterplot(
            data=subset,
            x=X_METRIC,
            y=Y_METRIC,
            hue='model',
            style='model_type',
            palette=pcfg.MODEL_COLORS,
            markers=SHAPE_MAPPING,
            hue_order=pcfg.MODEL_ORDER,
            style_order=SHAPE_ORDER,
            s=250,
            alpha=0.9,
            edgecolor='k',
            linewidth=0.5,
            ax=ax,
            legend=False
        )

        # Labels & Titles
        ax.set_title(f"{prog}{corr_str}", fontweight='bold', fontsize=13, pad=10)
        
        if i % cols == 0:
            ax.set_ylabel(r"Trajectory Fidelity ($\tau$)", labelpad=10)
        else:
            ax.set_ylabel("")
            
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Representation Granularity (Intrinsic Dim)", fontweight='bold', labelpad=10)
        else:
            ax.set_xlabel("")

        ax.grid(True, linestyle='--', alpha=0.3)
        sns.despine(ax=ax, trim=True)

    # 4. Legends (Reused logic)
    color_handles = [
        Line2D([0], [0], marker='o', color='w', label=pcfg.MODEL_LABELS.get(m, m),
               markerfacecolor=pcfg.MODEL_COLORS.get(m, 'k'), markersize=10, markeredgecolor='k')
        for m in pcfg.MODEL_ORDER
    ]
    shape_handles = [
        Line2D([0], [0], marker=SHAPE_MAPPING[m], color='w', label=m,
               markerfacecolor='#777777', markeredgecolor='k', markersize=12)
        for m in SHAPE_ORDER
    ]

    legend_shape = fig.legend(
        handles=shape_handles, loc='upper center', bbox_to_anchor=(0.5, 0.93), 
        ncol=3, frameon=False, fontsize=11, title="Model Types"
    )
    fig.legend(
        handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, 1.00), 
        ncol=6, frameon=False, fontsize=11, title="Specific Models"
    )
    fig.add_artist(legend_shape)

    plt.tight_layout(rect=[0, 0, 1, 0.88]) 
    
    save_path = OUTPUT_DIR / 'scatter_tau_vs_id_raw.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_id_vs_fidelity()