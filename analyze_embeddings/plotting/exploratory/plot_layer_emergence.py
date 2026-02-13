#!/usr/bin/env python
"""
Plotting Script: Layer-wise Metric Evolution.

Modes:
1. model_compare: Stack of metrics. X=Depth. Lines=Models. (Requires --type)
2. token_compare: Stack of metrics. X=Depth. Lines=Token Types. One file per model.

Updates:
  - Plots MEAN only (no CI shading).
  - Automatically skips empty metrics.
  - Handles missing registers dynamically.
  - FIX: Robust parsing of depth (ignores non-numeric rows).
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
pcfg.set_icml_style()

INPUT_FILE = config.FULL_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Maps CLI args to CSV substrings
PREFIX_MAP = {
    'patch_mean': 'patch_mean',
    'cls': 'cls',
    'register': 'register_mean'
}

# Reverse map for labeling in the new plots
TYPE_DISPLAY_MAP = {
    'patch_mean': 'Patch',
    'cls': 'CLS',
    'register_mean': 'Register'
}

# Consistent colors for token types
TOKEN_PALETTE = {
    'Patch': '#1f77b4',   # Blue
    'CLS': '#ff7f0e',     # Orange
    'Register': '#2ca02c' # Green
}

# Full list of potential metrics
METRICS = {
    'tau': r'Trajectory Fidelity ($\tau$)',
    'id_raw': 'Intrinsic Dim (Raw)',
    'id_diff': 'Intrinsic Dim (Diff)',
    'spectral_gap': 'Spectral Gap',
    'neighborhood_purity': 'Neighborhood Purity',
    'trustworthiness': 'Trustworthiness'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot metrics over layer depths.")
    
    parser.add_argument('--mode', type=str, default='model_compare',
                        choices=['model_compare', 'token_compare'],
                        help="Choose plotting mode: 'model_compare' (lines=models) or 'token_compare' (lines=tokens).")

    parser.add_argument('--type', type=str, required=False, 
                        choices=['patch_mean', 'cls', 'register'],
                        help="The embedding layer type (Required only for 'model_compare' mode).")
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == 'model_compare' and not args.type:
        parser.error("--type is required when mode is 'model_compare'")
        
    return args

def extract_depth_and_type(df):
    """
    Parses 'embedding_type' (e.g., 'patch_mean_0.125') into 'token_type' and 'depth'.
    Robustly handles rows that do not have a numeric depth suffix.
    """
    df = df.copy()
    
    # Split from right to handle underscores in names (patch_mean) vs depth
    split_data = df['embedding_type'].str.rsplit('_', n=1, expand=True)
    
    if split_data.shape[1] < 2:
        # Handle case where split fails entirely
        return pd.DataFrame() 

    df['token_type_raw'] = split_data[0]
    
    # FIX: Coerce to numeric (turns 'embedding' or other strings into NaN)
    df['depth'] = pd.to_numeric(split_data[1], errors='coerce')
    
    # Drop rows where depth failed to parse (i.e., non-layer-wise rows)
    df = df.dropna(subset=['depth'])
    
    return df

# -----------------------------------------------------------------------------
# Mode 1: Original (Compare Models for 1 Token Type)
# -----------------------------------------------------------------------------
def process_data_single_type(df, prefix):
    """Filters dataframe for the specific prefix."""
    mask = df['embedding_type'].str.startswith(f"{prefix}_")
    df_subset = df[mask].copy()

    if df_subset.empty:
        raise ValueError(f"No data found for prefix '{prefix}_'. Check your CSV.")

    df_subset = extract_depth_and_type(df_subset)
    
    # Enforce Model Ordering
    df_subset['model'] = pd.Categorical(
        df_subset['model'], 
        categories=pcfg.MODEL_ORDER, 
        ordered=True
    )
    
    return df_subset.sort_values(['model', 'depth'])

def plot_vertical_stack_by_model(df, layer_type):
    valid_metrics = [m for m in METRICS.keys() if m in df.columns and df[m].notna().any()]
    if not valid_metrics:
        print("Error: No valid metrics found to plot!")
        return

    num_metrics = len(valid_metrics)
    fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(8, 2.5 * num_metrics), sharex=True)
    if num_metrics == 1: axes = [axes]

    handles, labels = None, None

    for i, metric in enumerate(valid_metrics):
        ax = axes[i]
        sns.lineplot(
            data=df, x='depth', y=metric, 
            hue='model', style='model',
            palette=pcfg.MODEL_COLORS,
            markers=True, dashes=False, markersize=8, linewidth=2,
            errorbar=None, ax=ax, legend=(i == 0)
        )
        ax.set_ylabel(METRICS[metric], fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        if ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    axes[-1].set_xlabel("Network Depth (Normalized)", fontweight='bold')
    axes[-1].set_xticks([0.125, 0.375, 0.625, 0.875, 1.0])
    
    display_name = PREFIX_MAP[layer_type].replace('_', ' ').title()
    fig.suptitle(f"Layer-wise Evolution: {display_name}", fontsize=16, fontweight='bold', y=0.99)

    if handles and labels:
        # Reorder legend based on config
        hl_dict = dict(zip(labels, handles))
        ordered_handles = [hl_dict[m] for m in pcfg.MODEL_ORDER if m in hl_dict]
        ordered_labels = [pcfg.MODEL_LABELS.get(m, m) for m in pcfg.MODEL_ORDER if m in hl_dict]
        
        fig.legend(ordered_handles, ordered_labels, loc='upper center', 
                   bbox_to_anchor=(0.5, 0.97), ncol=len(ordered_labels), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = OUTPUT_DIR / f"metrics_depth_stack_{layer_type}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

# -----------------------------------------------------------------------------
# Mode 2: New (Compare Tokens for each Model)
# -----------------------------------------------------------------------------
def plot_vertical_stack_by_token(df):
    """
    Generates one stacked plot per MODEL.
    Lines represent TOKEN TYPES (Patch, CLS, Register).
    """
    # 1. Clean and parse depth first
    df = extract_depth_and_type(df)
    
    # 2. Map raw strings (patch_mean) to display strings (Patch)
    # This acts as a second filter for unwanted rows
    df = df[df['token_type_raw'].isin(TYPE_DISPLAY_MAP.keys())].copy()
    df['Token Type'] = df['token_type_raw'].map(TYPE_DISPLAY_MAP)

    unique_models = df['model'].unique()

    for model in unique_models:
        print(f"  > Generating plot for model: {model}...")
        model_df = df[df['model'] == model].copy()
        
        # Check which metrics exist for this specific model
        valid_metrics = [m for m in METRICS.keys() if m in model_df.columns and model_df[m].notna().any()]
        
        if not valid_metrics:
            print(f"    Skipping {model} (No valid metrics)")
            continue

        num_metrics = len(valid_metrics)
        fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(8, 2.5 * num_metrics), sharex=True)
        if num_metrics == 1: axes = [axes]

        handles, labels = None, None

        for i, metric in enumerate(valid_metrics):
            ax = axes[i]
            
            # Plot
            sns.lineplot(
                data=model_df, 
                x='depth', 
                y=metric, 
                hue='Token Type',
                style='Token Type',
                palette=TOKEN_PALETTE, # Ensures CLS is always Orange, etc.
                markers=True, 
                dashes=False, 
                markersize=8, 
                linewidth=2,
                errorbar=None,
                ax=ax,
                legend=(i == 0)
            )

            ax.set_ylabel(METRICS[metric], fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Grab legend from first plot only
            if ax.get_legend():
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

        # Formatting
        axes[-1].set_xlabel("Network Depth (Normalized)", fontweight='bold')
        axes[-1].set_xticks([0.125, 0.375, 0.625, 0.875, 1.0])
        
        model_display = pcfg.MODEL_LABELS.get(model, model)
        fig.suptitle(f"Token Emergence: {model_display}", fontsize=16, fontweight='bold', y=0.99)

        # Legend
        if handles and labels:
            fig.legend(handles, labels, loc='upper center', 
                       bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = OUTPUT_DIR / f"token_emergence_{model}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close to save memory in loop

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    
    print(f"Loading data from {INPUT_FILE}...")
    df_raw = pd.read_csv(INPUT_FILE)

    if args.mode == 'model_compare':
        prefix = PREFIX_MAP[args.type]
        print(f"Mode: Model Comparison (Layer Type: {prefix})")
        df_clean = process_data_single_type(df_raw, prefix)
        plot_vertical_stack_by_model(df_clean, args.type)
        
    elif args.mode == 'token_compare':
        print("Mode: Token Comparison (One plot per model)")
        plot_vertical_stack_by_token(df_raw)