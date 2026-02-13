#!/usr/bin/env python
"""
Plotting Script: Semantic Collapse Evidence (Pairwise AUROC).
Aggregates pairwise AUROC scores to show how well models separate adjacent stages.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# Import Root Config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

# Import Local Plotting Config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
pcfg.set_icml_style() # Apply Shared Style

OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Offsets for grouping bars around the tick mark
OFFSETS = {
    "BDC": -0.2,
    "CRC-Conventional": -0.07,
    "CRC-Serrated": 0.07,
    "SCC": 0.2
}

def load_and_prep_data():
    """
    Loads Real CSV and aggregates pairwise AUROC columns.
    Returns a dataframe with 'mean_auroc', 'mean_auroc_lower', 'mean_auroc_upper'.
    """
    try:
        df = pd.read_csv("full_manifold_evaluation_100_with_auroc.csv") # Updated filename
        df = df[df["embedding_type"] == "final_embedding"].copy()
    except FileNotFoundError:
        print(f"Error: Could not find results CSV.")
        sys.exit(1)

    # 1. Identify AUROC columns dynamically
    # Columns look like: 'auroc_Hyperplastic_vs_Sessile', 'auroc_Hyperplastic_vs_Sessile_ci_lower'
    auroc_cols = [c for c in df.columns if c.startswith("auroc_") and "_ci_" not in c]
    
    if not auroc_cols:
        print("Error: No 'auroc_' columns found in CSV. Did you run the updated evaluation script?")
        sys.exit(1)

    # 2. Calculate Row-wise Average AUROC (across all pairs for that specific run)
    # We want the average separability for that disease/model combo
    # Note: NaNs (if a pair was missing) are ignored by mean(axis=1)
    df["mean_auroc"] = df[auroc_cols].mean(axis=1)

    # 3. Calculate Aggregated CIs
    # Since we have CIs for each pair, approximating the CI of the mean is complex without the raw bootstraps.
    # PROXY: We will average the lower bounds and average the upper bounds. 
    # This is a conservative visual estimate for the plot.
    lower_cols = [f"{c}_ci_lower" for c in auroc_cols]
    upper_cols = [f"{c}_ci_upper" for c in auroc_cols]
    
    # Filter for columns that actually exist in the CSV (in case some pairs failed bootstrap)
    valid_lower = [c for c in lower_cols if c in df.columns]
    valid_upper = [c for c in upper_cols if c in df.columns]

    df["mean_auroc_lower"] = df[valid_lower].mean(axis=1)
    df["mean_auroc_upper"] = df[valid_upper].mean(axis=1)

    return df

def main():
    df = load_and_prep_data()
    
    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(15, 6)) 
    
    # Use Standardized Model Order
    x_base = np.arange(len(pcfg.MODEL_ORDER))
    
    # --- 2. Plot Data ---
    for prog in df["progression"].unique():
        if prog not in pcfg.PROGRESSION_COLORS: continue
        if prog == "Null": continue # No AUROC for Null usually
        
        subset = df[df["progression"] == prog].set_index("model")
        subset = subset.reindex(pcfg.MODEL_ORDER) # Enforce Order
        
        xs = x_base + OFFSETS[prog]
        ys = subset["mean_auroc"].values
        
        # Calculate error bars relative to the mean
        yerr = np.array([
            (subset["mean_auroc"] - subset["mean_auroc_lower"]).values,
            (subset["mean_auroc_upper"] - subset["mean_auroc"]).values
        ])
        
        mask = ~np.isnan(ys)
        
        # Plot Error Bars
        ax.errorbar(
            xs[mask], ys[mask], yerr=yerr[:, mask],
            fmt='o', 
            color=pcfg.PROGRESSION_COLORS[prog],
            ecolor=pcfg.PROGRESSION_COLORS[prog],
            capsize=3, elinewidth=1.5, alpha=0.9, markersize=6
        )

    # --- 3. Styling & Annotation ---
    xtick_labels = [pcfg.MODEL_LABELS.get(m, m) for m in pcfg.MODEL_ORDER]
    
    ax.set_xticks(x_base)
    ax.set_xticklabels(xtick_labels, fontweight='bold', fontsize=11)
    
    ax.set_ylabel("Avg. Pairwise AUROC (Separability)", fontsize=12)
    
    # Dashed line at 0.5 (Random Guess / Total Collapse)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(x_base[-1] + 0.5, 0.51, "Random/Collapse", fontsize=9, color='black', va='bottom', ha='right')

    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Clean Y-Axis limits
    # AUROC is 0.5 to 1.0 generally
    ax.set_ylim(bottom=0.45, top=1.0)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))

    # --- Background Regions ---
    trans = ax.get_xaxis_transform()
    label_y_pos = -0.14 
    
    # 1. Natural Image
    ax.axvspan(-0.5, 0.5, color='gray', alpha=0.05, lw=0)
    ax.text(0, label_y_pos, "Natural Image", ha='center', va='top',
            color='gray', alpha=0.6, fontweight='bold', transform=trans)

    # 2. Vision-Language
    ax.axvspan(0.5, 2.5, color='purple', alpha=0.05, lw=0)
    ax.text(1.5, label_y_pos, "Vision-Language (Pathology)", ha='center', va='top',
            color='purple', alpha=0.6, fontweight='bold', transform=trans)

    # 3. Pathology
    ax.axvspan(2.5, 5.5, color='blue', alpha=0.05, lw=0)
    ax.text(4.0, label_y_pos, "Vision-Only (Pathology)", ha='center', va='top',
            color='blue', alpha=0.6, fontweight='bold', transform=trans)

    # --- 4. Global Legend & Title ---
    
    fig.suptitle("Semantic Collapse: Vision-Language models struggle to separate adjacent stages", 
                 fontsize=14, fontweight='bold', y=0.98)

    # Create Legend Handles
    legend_handles = []
    for prog, color in pcfg.PROGRESSION_COLORS.items():
        if prog == "Null": continue
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                   markersize=9, label=prog)
        )

    fig.legend(handles=legend_handles, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.93), 
               ncol=4, 
               frameon=False,
               fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    output_filename = OUTPUT_DIR / "semantic_collapse_auroc.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    main()