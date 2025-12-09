#!/usr/bin/env python
"""
Plotting Script: Manifold Dynamics (3-Panel)
Location: plotting/plot_manifold_dynamics.py
Execution: Run from root via `python plotting/plot_manifold_dynamics.py`
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

# -----------------------------------------------------------------------------
# Imports & Setup
# -----------------------------------------------------------------------------
try:
    import plot_config
except ImportError:
    try:
        from plotting import plot_config
    except ImportError:
        sys.exit("Critical Error: Could not import 'plot_config'. Ensure it exists in the 'plotting/' directory.")

plot_config.set_icml_style()

# -----------------------------------------------------------------------------
# Config & Mappings
# -----------------------------------------------------------------------------
INPUT_FILE = "results/full_manifold_evaluation_100.csv"
OUTPUT_FILE = "plots/manifold_dynamics_tri_panel_icml.png"

METRIC_COLORS = {
    'id_raw': '#1f77b4',   # Blue
    'id_diff': '#d62728'   # Red
}

LAYER_MAPPING = {
    "dinov2":   {0.125: 2, 0.375: 8, 0.625: 14, 0.875: 20, 1.0: 23},
    "uni2":     {0.125: 2, 0.375: 8, 0.625: 14, 0.875: 20, 1.0: 23},
    "gigapath": {0.125: 4, 0.375: 14, 0.625: 24, 0.875: 34, 1.0: 39},
    "conch":    {0.125: 1, 0.375: 3, 0.625: 7, 0.875: 9, 1.0: 11},
    "musk":     {0.125: 2, 0.375: 8, 0.625: 14, 0.875: 20, 1.0: 23},
    "virchow2": {0.125: 3, 0.375: 11, 0.625: 19, 0.875: 27, 1.0: 31},
}

# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------
def get_absolute_layer(row):
    """Maps model name and depth pct to integer layer index."""
    model = row.get('model')
    depth = row.get('depth')
    mapping = LAYER_MAPPING.get(model, {})
    return mapping.get(depth, np.nan)

def load_and_process_data(filepath):
    if not os.path.exists(filepath):
        if os.path.exists(f"../{filepath}"):
            filepath = f"../{filepath}"
        else:
            print(f"Warning: {filepath} not found.")

    df = pd.read_csv(filepath)

    # 1. Parse Depth Percentage
    split_data = df['embedding_type'].str.rsplit('_', n=1, expand=True)
    if split_data.shape[1] > 1:
        df['depth'] = pd.to_numeric(split_data[1], errors='coerce')
    else:
        df['depth'] = np.nan
        
    df = df.dropna(subset=['depth'])

    # 2. Map Absolute Layer Index
    df['abs_layer'] = df.apply(get_absolute_layer, axis=1)

    # 3. Enforce Categorical Ordering
    existing_models = [m for m in plot_config.MODEL_ORDER if m in df['model'].unique()]
    df['model'] = pd.Categorical(df['model'], categories=existing_models, ordered=True)

    return df

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def format_legend_labels(ax):
    """Updates legend text to use pretty names from plot_config."""
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [plot_config.MODEL_LABELS.get(label, label) for label in labels]
    ax.legend(handles, new_labels, title=None, frameon=False, loc='upper left')

def plot_tri_panel(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    # === PANEL A: Trajectory Fidelity (Tau) vs Depth % ===
    ax_tau = axes[0]
    sns.lineplot(
        data=df, x='depth', y='tau', hue='model',
        palette=plot_config.MODEL_COLORS, 
        hue_order=plot_config.MODEL_ORDER,
        linewidth=2, marker='o', 
        ax=ax_tau, errorbar=None  # REMOVED CI BARS HERE
    )
    
    ax_tau.set_title("Emergence of Trajectory Fidelity", fontweight='bold')
    ax_tau.set_ylabel(r"Trajectory Fidelity ($\tau$)")
    ax_tau.set_xlabel("Network Depth (%)")
    ax_tau.set_ylim(0, 0.8)
    
    ax_tau.set_xticks([0.125, 0.375, 0.625, 0.875, 1.0])
    ax_tau.set_xticklabels(["12.5", "37.5", "62.5", "87.5", "100"])
    
    format_legend_labels(ax_tau)
    
    # === PANEL B: Manifold Stability (Aggregated) ===
    # I left CI bars here as this shows global variance across models, 
    # but let me know if you want them removed here too.
    ax_id = axes[1]
    
    # Feature Complexity
    sns.lineplot(
        data=df, x='depth', y='id_raw',
        color=METRIC_COLORS['id_raw'], linewidth=2.5, marker='o', markersize=7,
        label='Embedding Space', ax=ax_id, errorbar=('ci', 95)
    )

    # Manifold Stability
    sns.lineplot(
        data=df, x='depth', y='id_diff',
        color=METRIC_COLORS['id_diff'], linewidth=2.5, marker='s', markersize=7,
        label='Diffusion Space', ax=ax_id, errorbar=('ci', 95)
    )

    ax_id.set_title("Representation Complexity (ID)", fontweight='bold')
    ax_id.set_ylabel("Intrinsic Dimensionality")
    ax_id.set_xlabel("Network Depth (%)")
    ax_id.set_ylim(0, 26)
    
    ax_id.set_xticks([0.125, 0.375, 0.625, 0.875, 1.0])
    ax_id.set_xticklabels(["12.5", "37.5", "62.5", "87.5", "100"])
    
    ax_id.legend(frameon=False)

    # === PANEL C: Trajectory Fidelity vs Absolute Layer ===
    ax_abs = axes[2]
    sns.lineplot(
        data=df, x='abs_layer', y='tau', hue='model',
        palette=plot_config.MODEL_COLORS, 
        hue_order=plot_config.MODEL_ORDER,
        linewidth=2, marker='D', markersize=6,
        ax=ax_abs, errorbar=None, legend=False  # REMOVED CI BARS HERE
    )
    
    ax_abs.set_title("Fidelity by Absolute Layer", fontweight='bold')
    ax_abs.set_ylabel(r"Trajectory Fidelity ($\tau$)")
    ax_abs.set_xlabel("Transformer Block Index")
    ax_abs.set_ylim(0, 0.8)
    
    ax_abs.set_xticks(np.arange(0, 41, 10))

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_FILE}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading data from {INPUT_FILE}...")
    df_clean = load_and_process_data(INPUT_FILE)
    
    if df_clean['abs_layer'].isnull().any():
         print("Warning: Some rows could not map depth % to absolute layer. Check JSON mapping.")
    
    plot_tri_panel(df_clean)