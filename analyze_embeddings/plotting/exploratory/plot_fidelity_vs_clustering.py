#!/usr/bin/env python
"""
Plotting Script: Fidelity (Tau) vs. Clustering (Silhouette).
Generates 4 subplots in a 2x2 grid.
Colors by specific model (from config), SHAPES by model category.
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
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration & Mappings
# -----------------------------------------------------------------------------
pcfg.set_icml_style()

INPUT_FILE = config.FULL_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EMBEDDING = 'final_embedding'

# --- 1. Define Model Categories & Shapes ---

# Define distinct shapes for these categories
SHAPE_MAPPING = {
    'DINOv2 Baseline': 'D',      # Diamond
    'Vision-Language (Path)': 'X',      # X mark
    'Vision-Only (Path)': 'o'    # Circle
}

# Define order for consistency in plotting and legend
SHAPE_ORDER = ['Vision-Only (Path)', 'Vision-Language (Path)', 'DINOv2 Baseline']


def plot_scatter_grid_with_shapes():
    # 2. Load Initial Data
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # Filter for Final Embedding & Config Models Only
    df = df[df['embedding_type'] == TARGET_EMBEDDING].copy()
    df = df[df['model'].isin(pcfg.MODEL_ORDER)]
    
    # Enforce Categorical Order for models
    df['model'] = pd.Categorical(
        df['model'], 
        categories=pcfg.MODEL_ORDER, 
        ordered=True
    )
    
    # --- Map Models to Categories for Shapes ---
    # Create the new column defining the broader category type
    df['model_type'] = df['model'].map(pcfg.MODEL_TYPE_MAPPING)
    
    df = df.sort_values(['model_type', 'model'])

    # Identify the 4 specific diseases from config (excluding 'Null')
    target_progs = [p for p in pcfg.PROGRESSION_COLORS.keys() if p != 'Null']

    # 3. Setup Grid (2x2)
    cols = 2
    rows = 2
    
    # Using sharex=False as silhouette ranges might vary slightly, sharey=True for direct tau comparison
    fig, axes = plt.subplots(rows, cols, figsize=(12, 11), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    # 4. Loop through Diseases
    for i, prog in enumerate(target_progs):
        if i >= len(axes_flat): break
        
        ax = axes_flat[i]
        
        # Subset data for this specific disease
        subset = df[df['progression'] == prog].copy()
        
        # Calculate Correlation based on valid pairs
        valid_data = subset.dropna(subset=['silhouette', 'tau'])
        corr_str = ""
        if len(valid_data) > 2: # Need at least 3 points for meaningful correlation
            r, p_val = pearsonr(valid_data['silhouette'], valid_data['tau'])
            corr_str = f"\n(Pearson r={r:.2f})"
        
        # --- Scatter Plot ---
        sns.scatterplot(
            data=subset,
            x='silhouette',
            y='tau',
            hue='model',           # Color by specific model
            style='model_type',    # Shape by broader category
            palette=pcfg.MODEL_COLORS,
            markers=SHAPE_MAPPING, # Use custom shapes defined above
            hue_order=pcfg.MODEL_ORDER,
            style_order=SHAPE_ORDER,
            s=250,                 # Size
            alpha=0.9,
            edgecolor='k',
            linewidth=0.5,
            ax=ax,
            legend=False           # Building global legends manually below
        )

        # --- Formatting ---
        ax.set_title(f"{prog}{corr_str}", fontweight='bold', fontsize=13, pad=10)
        
        # Y-Label only on left columns
        if i % cols == 0:
            ax.set_ylabel(r"Trajectory Fidelity ($\tau$)", labelpad=10)
        else:
            ax.set_ylabel("")
            
        # X-Label only on bottom rows
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Clustering Quality (Silhouette)", fontweight='bold', labelpad=10)
        else:
            ax.set_xlabel("")

        ax.grid(True, linestyle='--', alpha=0.3)
        sns.despine(ax=ax, trim=True)

    # --- 5. Global Legends ---
    
    # A) Color Legend (Specific Models)
    color_legend_handles = []
    for model in pcfg.MODEL_ORDER:
        label = pcfg.MODEL_LABELS.get(model, model)
        color = pcfg.MODEL_COLORS.get(model, 'black')
        # Use a generic circle for the color legend entries
        h = Line2D([0], [0], marker='o', color='w', label=label,
                   markerfacecolor=color, markersize=10, markeredgecolor='k')
        color_legend_handles.append(h)

    # B) Shape Legend (Model Categories)
    shape_legend_handles = []
    for m_type in SHAPE_ORDER:
        shape = SHAPE_MAPPING[m_type]
        # Use neutral grey for the shape legend entries to emphasize form over color
        h = Line2D([0], [0], marker=shape, color='w', label=m_type,
                   markerfacecolor='#777777', markeredgecolor='k', markersize=12)
        shape_legend_handles.append(h)

    # Add legends to figure stacked at the top
    # 1. Add Shape Legend (lower stack)
    legend_shape = fig.legend(handles=shape_legend_handles, loc='upper center', 
               bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=11, title="Model Types (Shapes)")
    
    # 2. Add Color Legend (upper stack) - Add this second so it's "on top" in z-order if they overlap
    fig.legend(handles=color_legend_handles, loc='upper center', 
               bbox_to_anchor=(0.5, 1.00), ncol=6, frameon=False, fontsize=11, title="Specific Models (Colors)")

    # Make sure the first legend isn't overwritten by the second call
    fig.add_artist(legend_shape)

    # 6. Global Title & Save
    # Adjust layout to leave room for double legend stack at top
    plt.tight_layout(rect=[0, 0, 1, 0.88]) 
    
    save_path = OUTPUT_DIR / 'scatter_tau_vs_silhouette_shapes.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_scatter_grid_with_shapes()