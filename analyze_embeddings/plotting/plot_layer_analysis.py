#!/usr/bin/env python
"""
Plotting Script: Emergence of Trajectory Fidelity
Location: plotting/plot_emergence.py

Generates:
1. Stacked vertical figure (2 panels) for main text: emergence + dimensionality (patch_mean tokens)
2. Three-panel figures for appendix: patch_mean, cls, and register_mean token analysis
   (each showing: relative depth tau, ID divergence, absolute depth tau)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Imports & Setup
# -----------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg

pcfg.set_icml_style()

# -----------------------------------------------------------------------------
# Config & Mappings
# -----------------------------------------------------------------------------
INPUT_FILE = config.FULL_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR / "emergence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metric colors for dimensionality panel
METRIC_COLORS = {
    'id_raw': '#1f77b4',   # Blue - Embedding Space
    'id_diff': '#d62728'   # Red - Diffusion Space
}

# Mapping from relative depth to absolute layer index per model
LAYER_MAPPING = {
    "dinov2":   {0.125: 2, 0.375: 8, 0.625: 14, 0.875: 20, 1.0: 23},
    "uni2":     {0.125: 2, 0.375: 8, 0.625: 14, 0.875: 20, 1.0: 23},
    "gigapath": {0.125: 4, 0.375: 14, 0.625: 24, 0.875: 34, 1.0: 39},
    "conch":    {0.125: 1, 0.375: 3, 0.625: 7, 0.875: 9, 1.0: 11},
    "musk":     {0.125: 2, 0.375: 8, 0.625: 14, 0.875: 20, 1.0: 23},
    "virchow2": {0.125: 3, 0.375: 11, 0.625: 19, 0.875: 27, 1.0: 31},
}

# Token type prefixes and display names
TOKEN_TYPES = {
    'cls': 'cls_',
    'patch_mean': 'patch_mean_',
    'register_mean': 'register_mean_'
}

TOKEN_DISPLAY_NAMES = {
    'cls': 'CLS Token',
    'patch_mean': 'Patch Mean',
    'register_mean': 'Register Mean'
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


def load_and_process_data(filepath, token_type='cls'):
    """Load and process the manifold evaluation data for a specific token type.
    
    Args:
        filepath: Path to CSV file
        token_type: One of 'cls', 'patch_mean', 'register_mean'
    """
    df = pd.read_csv(filepath)
    
    # Filter by token type prefix
    prefix = TOKEN_TYPES.get(token_type, 'cls_')
    df = df[df['embedding_type'].str.startswith(prefix)].copy()

    # Parse Depth Percentage from embedding_type column
    split_data = df['embedding_type'].str.rsplit('_', n=1, expand=True)
    if split_data.shape[1] > 1:
        df['depth'] = pd.to_numeric(split_data[1], errors='coerce')
    else:
        df['depth'] = np.nan
        
    df = df.dropna(subset=['depth'])

    # Map Absolute Layer Index
    df['abs_layer'] = df.apply(get_absolute_layer, axis=1)

    # Enforce Categorical Ordering
    existing_models = [m for m in pcfg.MODEL_ORDER if m in df['model'].unique()]
    df['model'] = pd.Categorical(df['model'], categories=existing_models, ordered=True)

    return df


def format_legend_labels(ax, loc='upper left'):
    """Updates legend text to use pretty names from plot_config."""
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [pcfg.MODEL_LABELS.get(label, label) for label in labels]
    ax.legend(handles, new_labels, title=None, frameon=False, loc=loc,
              fontsize=pcfg.FONT_SIZES['legend'])


# -----------------------------------------------------------------------------
# Main Text Figure: Stacked Vertical (Emergence + Dimensionality) - Patch Mean
# -----------------------------------------------------------------------------
def plot_main_figure(df, token_type='patch_mean'):
    """Generate stacked vertical figure for main text using patch_mean."""
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)
    
    # === TOP PANEL: Trajectory Fidelity vs Relative Depth ===
    ax_tau = axes[0]
    
    for model in pcfg.MODEL_ORDER:
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue
        
        # Average across progressions for each depth
        avg_df = model_df.groupby('depth')['tau'].mean().reset_index()
        
        ax_tau.plot(
            avg_df['depth'], avg_df['tau'],
            color=pcfg.MODEL_COLORS[model],
            marker=pcfg.MODEL_MARKERS[model],
            linewidth=2,
            markersize=7,
            label=model
        )
    
    ax_tau.set_ylabel(r"Trajectory Fidelity ($\tau$)", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_tau.set_xlabel("")  # Remove x-axis label from top panel
    ax_tau.set_ylim(0, 0.8)
    
    ax_tau.set_xticks([0.125, 0.375, 0.625, 0.875, 1.0])
    ax_tau.set_xticklabels(["12.5", "37.5", "62.5", "87.5", "100"])
    
    ax_tau.yaxis.grid(True, linestyle='--', alpha=0.3)
    format_legend_labels(ax_tau, loc='upper left')
    
    sns.despine(ax=ax_tau)

    # === BOTTOM PANEL: Intrinsic Dimensionality ===
    ax_id = axes[1]
    
    # Embedding Space ID (aggregated across models)
    sns.lineplot(
        data=df, x='depth', y='id_raw',
        color=METRIC_COLORS['id_raw'], 
        linewidth=2.5, 
        marker='o', 
        markersize=7,
        label='Embedding Space', 
        ax=ax_id, 
        errorbar=('ci', 95)
    )

    # Diffusion Space ID (aggregated across models)
    sns.lineplot(
        data=df, x='depth', y='id_diff',
        color=METRIC_COLORS['id_diff'], 
        linewidth=2.5, 
        marker='s', 
        markersize=7,
        label='Diffusion Space', 
        ax=ax_id, 
        errorbar=('ci', 95)
    )

    ax_id.set_ylabel("Intrinsic Dimensionality", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_id.set_xlabel("Network Depth (%)", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_id.set_ylim(0, 25)
    
    ax_id.set_xticks([0.125, 0.375, 0.625, 0.875, 1.0])
    ax_id.set_xticklabels(["12.5", "37.5", "62.5", "87.5", "100"])
    
    ax_id.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_id.legend(frameon=False, loc='upper left', fontsize=pcfg.FONT_SIZES['legend'])
    
    sns.despine(ax=ax_id)

    # Save
    output_path = OUTPUT_DIR / f"emergence_stacked_{token_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Main figure saved to {output_path}")


# -----------------------------------------------------------------------------
# Appendix Figure: Three-Panel (for all token types)
# Shows: (1) Tau vs relative depth, (2) ID divergence, (3) Tau vs absolute depth
# -----------------------------------------------------------------------------
def plot_three_panel_figure(df, token_type='patch_mean'):
    """Generate three-panel figure for appendix."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    
    display_name = TOKEN_DISPLAY_NAMES.get(token_type, token_type)
    
    # === PANEL A: Trajectory Fidelity vs Relative Depth ===
    ax_tau = axes[0]
    
    for model in pcfg.MODEL_ORDER:
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue
        
        avg_df = model_df.groupby('depth')['tau'].mean().reset_index()
        
        ax_tau.plot(
            avg_df['depth'], avg_df['tau'],
            color=pcfg.MODEL_COLORS[model],
            marker=pcfg.MODEL_MARKERS[model],
            linewidth=2,
            markersize=7,
            label=model
        )
    
    ax_tau.set_ylabel(r"Trajectory Fidelity ($\tau$)", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_tau.set_xlabel("Network Depth (%)", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_tau.set_ylim(0, 0.8)
    
    ax_tau.set_xticks([0.125, 0.375, 0.625, 0.875, 1.0])
    ax_tau.set_xticklabels(["12.5", "37.5", "62.5", "87.5", "100"])
    
    ax_tau.yaxis.grid(True, linestyle='--', alpha=0.3)
    format_legend_labels(ax_tau, loc='upper left')
    sns.despine(ax=ax_tau)

    # === PANEL B: Intrinsic Dimensionality ===
    ax_id = axes[1]
    
    sns.lineplot(
        data=df, x='depth', y='id_raw',
        color=METRIC_COLORS['id_raw'], 
        linewidth=2.5, 
        marker='o', 
        markersize=7,
        label='Embedding Space', 
        ax=ax_id, 
        errorbar=('ci', 95)
    )

    sns.lineplot(
        data=df, x='depth', y='id_diff',
        color=METRIC_COLORS['id_diff'], 
        linewidth=2.5, 
        marker='s', 
        markersize=7,
        label='Diffusion Space', 
        ax=ax_id, 
        errorbar=('ci', 95)
    )

    ax_id.set_ylabel("Intrinsic Dimensionality", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_id.set_xlabel("Network Depth (%)", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_id.set_ylim(0, 25)
    
    ax_id.set_xticks([0.125, 0.375, 0.625, 0.875, 1.0])
    ax_id.set_xticklabels(["12.5", "37.5", "62.5", "87.5", "100"])
    
    ax_id.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_id.legend(frameon=False, loc='upper left', fontsize=pcfg.FONT_SIZES['legend'])
    sns.despine(ax=ax_id)

    # === PANEL C: Trajectory Fidelity vs Absolute Layer ===
    ax_abs = axes[2]
    
    for model in pcfg.MODEL_ORDER:
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue
        
        avg_df = model_df.groupby('abs_layer')['tau'].mean().reset_index()
        
        ax_abs.plot(
            avg_df['abs_layer'], avg_df['tau'],
            color=pcfg.MODEL_COLORS[model],
            marker=pcfg.MODEL_MARKERS[model],
            linewidth=2,
            markersize=7,
            label=model
        )
    
    ax_abs.set_ylabel(r"Trajectory Fidelity ($\tau$)", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_abs.set_xlabel("Transformer Block Index", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax_abs.set_ylim(0, 0.8)
    ax_abs.set_xticks(np.arange(0, 41, 10))
    
    ax_abs.yaxis.grid(True, linestyle='--', alpha=0.3)
    # No legend on third panel to avoid redundancy
    sns.despine(ax=ax_abs)

    # Save
    output_path = OUTPUT_DIR / f"emergence_three_panel_{token_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Three-panel figure ({display_name}) saved to {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading data from {INPUT_FILE}...")
    
    # ==========================================================================
    # MAIN TEXT: Patch mean stacked figure
    # ==========================================================================
    print("\n=== Patch Mean Analysis (Main Text) ===")
    df_patch = load_and_process_data(INPUT_FILE, token_type='patch_mean')
    
    if df_patch['abs_layer'].isnull().any():
        print("Warning: Some rows could not map depth % to absolute layer. Check LAYER_MAPPING.")
    
    print("Generating main text figure (stacked, patch_mean)...")
    plot_main_figure(df_patch, token_type='patch_mean')

    # ==========================================================================
    # MAIN TEXT: CLS token stacked figure 
    # ==========================================================================
    print("\n=== CLS Token Analysis (Main Text) ===")
    df_cls = load_and_process_data(INPUT_FILE, token_type='cls')

    if df_cls['abs_layer'].isnull().any():
        print("Warning: Some rows could not map depth % to absolute layer. Check LAYER_MAPPING.")

    print("Generating main text figure (stacked, cls)...")
    plot_main_figure(df_cls, token_type='cls')
    
    # ==========================================================================
    # APPENDIX: Three-panel figures for all token types
    # ==========================================================================
    for token_type in ['patch_mean', 'cls', 'register_mean']:
        print(f"\n=== {TOKEN_DISPLAY_NAMES[token_type]} Analysis (Appendix) ===")
        df = load_and_process_data(INPUT_FILE, token_type=token_type)
        
        if df.empty:
            print(f"Warning: No data found for {token_type}, skipping...")
            continue
            
        print(f"Generating three-panel figure ({token_type})...")
        plot_three_panel_figure(df, token_type=token_type)
    
    print("\n✅ All figures generated successfully!")