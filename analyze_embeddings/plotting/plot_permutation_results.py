#!/usr/bin/env python
"""
Plotting Script: Stage Permutation Specificity.

Generates a Faceted Plot (one subplot per progression) comparing:
1. The Distribution of Permuted Taus (Boxplot = Null Hypothesis)
2. The Canonical Tau (Star Marker = Biological Signal)
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_FILE = config.PERMUTATION_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define Colors for Models (matching your previous schemes if possible)
# Or we can just use a standard palette.
MODEL_ORDER = config.EXPECTED_MODELS

# -----------------------------------------------------------------------------
# Plotting Logic
# -----------------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        print("   Run 'analysis/evaluate_stage_permutations.py' first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    
    # Separate Canonical and Permuted for easier plotting layers
    df_null = df[df["type"] == "permuted"].copy()
    df_real = df[df["type"] == "canonical"].copy()

    # Setup Seaborn Theme for Paper
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Create FacetGrid: One column per Progression
    # We use 'sharey=False' if scales differ wildly, but usually Tau is -1 to 1.
    # Keeping sharey=True makes them comparable.
    g = sns.FacetGrid(
        df, 
        col="progression", 
        col_wrap=2, 
        height=4, 
        aspect=1.5, 
        sharex=False,
        sharey=True 
    )

    # 1. Plot the Null Distributions (Boxplots)
    # We map using the full dataset but filter internally or pass df_null
    # Note: We must ensure x-order is consistent
    g.map_dataframe(
        sns.boxplot,
        x="model",
        y="tau",
        data=df_null, # Explicitly plot only nulls here
        order=MODEL_ORDER,
        color="#bdc3c7", # Neutral Gray
        linewidth=1.0,
        fliersize=0,     # Hide outliers to avoid confusion with the real point
        width=0.5,
        zorder=1
    )

    # 2. Plot the Individual Permutation Points (Jittered)
    # This adds transparency to show the raw data points behind the box
    g.map_dataframe(
        sns.stripplot,
        x="model",
        y="tau",
        data=df_null,
        order=MODEL_ORDER,
        color="#7f8c8d",
        alpha=0.4,
        size=3,
        jitter=True,
        zorder=1
    )

    # 3. Plot the Canonical Signal (Stars)
    # We use a distinct color palette for the models or just a high-contrast color (Red/Blue)
    g.map_dataframe(
        sns.stripplot,
        x="model",
        y="tau",
        data=df_real,
        order=MODEL_ORDER,
        palette="tab10", # Color by model
        marker="*",      # Star shape
        size=15,         # Large size
        edgecolor="black",
        linewidth=1,
        jitter=False,
        zorder=10        # Ensure it sits on top
    )

    # 4. Customization & Polish
    
    # Add a dashed line at 0 for reference
    for ax in g.axes.flat:
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3, zorder=0)

    # Titles and Labels
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Kendall's Tau")
    
    # Rotate X-axis labels for readability
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    # Add a custom legend manually (since we mixed plots)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Biological Order (Canonical)',
               markerfacecolor='black', markersize=15, markeredgecolor='black'),
        Line2D([0], [0], color='#bdc3c7', lw=4, label='Random Permutations (Null)'),
    ]
    
    # Place legend on the figure (top center or bottom)
    # Adjust bbox_to_anchor to fit your layout
    g.figure.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.05), 
        ncol=2, 
        frameon=False
    )

    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "permutation_specificity_test.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")

if __name__ == "__main__":
    main()