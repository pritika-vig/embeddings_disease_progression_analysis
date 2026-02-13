#!/usr/bin/env python
"""
Analysis Script: Architecture vs. Performance Correlation (v7).
Updates: Removes LaTeX formatting for plain text output and standard plot labels.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, pointbiserialr

# Import Root Config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration & Metadata
# -----------------------------------------------------------------------------
pcfg.set_icml_style()
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "DINOv2":    {"params": 304,  "dim": 1024, "blocks": 24, "type": "Natural Image", "registers": 4},
    "UNI-2":     {"params": 304,  "dim": 1536, "blocks": 24, "type": "Vision",        "registers": 8}, 
    "GigaPath":  {"params": 1100, "dim": 1536, "blocks": 40, "type": "Vision",        "registers": 0}, 
    "CONCH":     {"params": 200,  "dim": 512,  "blocks": 12, "type": "VL",            "registers": 0},
    "MuSK":      {"params": 625,  "dim": 1024, "blocks": 24, "type": "VL",            "registers": 0},
    "Virchow-2": {"params": 632,  "dim": 1280, "blocks": 32, "type": "Vision",        "registers": 4},
}

def load_and_aggregate_data():
    """Loads results and calculates mean Tau per model."""
    csv_path = Path("full_manifold_evaluation_100.csv")
    
    try:
        df = pd.read_csv(csv_path)
        df = df[df["embedding_type"] == "final_embedding"].copy()
    except FileNotFoundError:
        print(f"Error: Could not find results at {csv_path}")
        sys.exit(1)

    # 1. Normalize Names
    NAME_MAP = {
        "uni2": "UNI-2",
        "virchow2": "Virchow-2",
        "gigapath": "GigaPath",
        "conch": "CONCH",
        "musk": "MuSK",
        "dinov2": "DINOv2"
    }
    df["model"] = df["model"].replace(NAME_MAP)

    # 2. Filter
    df_filtered = df[df["model"].isin(MODEL_SPECS.keys())].copy()
    
    if df_filtered.empty:
        print("❌ ERROR: No data remaining after filtering!")
        sys.exit(1)

    # 3. Aggregate
    agg_df = df_filtered.groupby("model")["tau"].mean().reset_index()
    
    # 4. Enrich with metadata
    agg_df["params"] = agg_df["model"].apply(lambda x: MODEL_SPECS[x]["params"])
    agg_df["dim"] = agg_df["model"].apply(lambda x: MODEL_SPECS[x]["dim"])
    agg_df["type"] = agg_df["model"].apply(lambda x: MODEL_SPECS[x]["type"])
    
    # Binary Features
    agg_df["is_vl"] = agg_df["type"].apply(lambda x: 1 if x == "VL" else 0)
    agg_df["num_registers"] = agg_df["model"].apply(lambda x: MODEL_SPECS[x]["registers"])
    agg_df["has_registers"] = agg_df["num_registers"] > 0
    agg_df["has_registers_int"] = agg_df["has_registers"].astype(int)
    
    return agg_df

def plot_continuous_correlation(ax, df_pathology, df_all, x_col, y_col, xlabel, color_map):
    """Plots and returns (corr, p_val)."""
    if len(df_pathology) < 2: return np.nan, np.nan

    # 1. Stats (Pathology Only)
    corr, p_val = spearmanr(df_pathology[x_col], df_pathology[y_col])
    
    # 2. Plotting
    sns.regplot(
        data=df_pathology, x=x_col, y=y_col, 
        scatter=False, ax=ax, color="gray", 
        line_kws={"alpha": 0.4, "linestyle": "--"}, ci=None
    )
    sns.scatterplot(
        data=df_all, x=x_col, y=y_col, 
        hue="type", palette=color_map, s=150, ax=ax, edgecolor="k", zorder=5
    )
    
    for _, row in df_all.iterrows():
        ax.text(row[x_col], row[y_col] + 0.02, pcfg.MODEL_LABELS.get(row["model"], row["model"]), 
                fontsize=9, ha='center', alpha=0.8)

    # Text without LaTeX
    stats_text = f"Pathology Spearman\nrho = {corr:.2f} (p={p_val:.2f})"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    return corr, p_val

def plot_categorical_comparison(ax, df_pathology, df_all, x_col, y_col, xlabel, color_map, x_labels=None):
    """Plots and returns (corr, p_val)."""
    if len(df_pathology) < 2: return np.nan, np.nan

    # 1. Stats (Pathology Only)
    binary_col = x_col if df_pathology[x_col].dtype != bool else x_col + "_int"
    if binary_col not in df_pathology.columns: binary_col = x_col 
    
    corr, p_val = pointbiserialr(df_pathology[binary_col], df_pathology[y_col])
    
    # 2. Plotting
    sns.boxplot(data=df_pathology, x=x_col, y=y_col, ax=ax, width=0.4, palette="pastel", fliersize=0)
    sns.stripplot(data=df_all, x=x_col, y=y_col, hue="type", palette=color_map, 
                  size=10, ax=ax, jitter=True, edgecolor="k", linewidth=1, zorder=5)
    
    if x_labels: ax.set_xticklabels(x_labels)
        
    for i, row in df_all.iterrows():
        x_pos = 0 if row[x_col] == 0 or row[x_col] == False else 1
        ax.text(x_pos, row[y_col] + 0.015, pcfg.MODEL_LABELS.get(row["model"], row["model"]), 
                fontsize=9, ha='center', alpha=0.8)

    # Text without LaTeX
    stats_text = f"Pathology Point-Biserial\nr = {corr:.2f} (p={p_val:.2f})"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    if ax.legend_: ax.legend_.remove()
    return corr, p_val

def print_summary_table(stats_results):
    """Prints a standard text table."""
    print("\n" + "="*90)
    print("STATISTICAL SUMMARY (Pathology Models Only, N=5)")
    print("DINOv2 excluded from statistics as out-of-domain baseline.")
    print("="*90)
    
    # Header
    print(f"{'Feature':<25} | {'Metric':<20} | {'Coeff':<8} | {'p-value':<8} | {'Significance'}")
    print("-" * 90)
    
    for row in stats_results:
        sig = ""
        if row['p'] < 0.05: sig = "*"
        if row['p'] < 0.01: sig = "**"
        if row['p'] < 0.001: sig = "***"
        
        # Clean names for display
        metric_name = "Spearman Rho" if "Spearman" in row['metric'] else "Point-Biserial r"
        
        print(f"{row['feature']:<25} | {metric_name:<20} | {row['val']:>6.2f}   | {row['p']:>6.2f}   | {sig}")
        
    print("-" * 90)
    print("* p<0.05, ** p<0.01, *** p<0.001")
    print("="*90 + "\n")

def main():
    df_all = load_and_aggregate_data()
    df_pathology = df_all[df_all["type"] != "Natural Image"].copy()
    
    type_colors = {"Vision": "#2ca02c", "VL": "#9467bd", "Natural Image": "#7f7f7f"}
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)

    # Store stats for table
    stats_results = []

    # 1. Params
    r, p = plot_continuous_correlation(
        axes[0], df_pathology, df_all, "params", "tau", "Model Parameters (M)", type_colors
    )
    axes[0].set_ylabel("Mean Trajectory Fidelity (Tau)", fontsize=12, fontweight='bold')
    stats_results.append({"feature": "Parameter Count", "metric": "Spearman", "val": r, "p": p})

    # 2. Dimensions
    r, p = plot_continuous_correlation(
        axes[1], df_pathology, df_all, "dim", "tau", "Embedding Dimension", type_colors
    )
    axes[1].set_ylabel("")
    stats_results.append({"feature": "Embedding Dimension", "metric": "Spearman", "val": r, "p": p})

    # 3. Vision vs. VL
    r, p = plot_categorical_comparison(
        axes[2], df_pathology, df_pathology, "is_vl", "tau", "Pathology Model Modality", type_colors,
        x_labels=["Vision Only", "Vision-Language"]
    )
    axes[2].set_ylabel("")
    stats_results.append({"feature": "Modality (VL=1)", "metric": "Point-Biserial", "val": r, "p": p})

    # 4. Registers
    r, p = plot_categorical_comparison(
        axes[3], df_pathology, df_all, "has_registers", "tau", "Vision Registers", type_colors,
        x_labels=["No Registers", "Has Registers"]
    )
    axes[3].set_ylabel("")
    stats_results.append({"feature": "Vision Registers", "metric": "Point-Biserial", "val": r, "p": p})

    fig.suptitle("Architectural Drivers of Biological Progression Learning (Pathology Models Only)", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = OUTPUT_DIR / "architecture_correlations_v7_plain_text.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ Plot saved to {output_path}")
    
    # Print the Table
    print_summary_table(stats_results)

if __name__ == "__main__":
    main()