#!/usr/bin/env python
"""
Plotting Script: Stage Permutation Specificity (Averaged Across Progressions).
Single-column ICML figure showing pooled null distributions vs. true ordering.
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

PERM_FILE = config.PERMUTATION_RESULTS_OUTPUT_PATH
REAL_FILE = config.FULL_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EMBEDDING = 'final_embedding'

# Color for the real/biological ordering stars
STAR_COLOR = '#c0392b'  # Deep red - stands out against gray

def plot_specificity_averaged():
    # 1. Load Data
    df_real = pd.read_csv(REAL_FILE)
    df_perm = pd.read_csv(PERM_FILE)

    # 2. Filter Data
    df_noise = df_perm[df_perm['type'] == 'permuted'].copy()
    df_signal = df_real[df_real['embedding_type'] == TARGET_EMBEDDING].copy()

    # --- Enforce Shared Model Order ---
    df_noise['model'] = pd.Categorical(df_noise['model'], categories=pcfg.MODEL_ORDER, ordered=True)
    df_signal['model'] = pd.Categorical(df_signal['model'], categories=pcfg.MODEL_ORDER, ordered=True)

    # 3. Pool null distributions across progressions for each model
    df_noise_pooled = df_noise.groupby(['model']).apply(
        lambda x: x, include_groups=False
    ).reset_index(drop=True)

    # 4. Compute mean and CI across progressions for real values
    df_signal_avg = df_signal.groupby('model').agg(
        tau_mean=('tau', 'mean'),
        tau_std=('tau', 'std'),
        tau_count=('tau', 'count'),
        tau_ci_lower=('tau_ci_lower', 'mean'),
        tau_ci_upper=('tau_ci_upper', 'mean'),
    ).reset_index()
    
    # Compute 95% CI for the mean across progressions
    df_signal_avg['tau_se'] = df_signal_avg['tau_std'] / np.sqrt(df_signal_avg['tau_count'])
    df_signal_avg['tau_ci_lower_pooled'] = df_signal_avg['tau_mean'] - 1.96 * df_signal_avg['tau_se']
    df_signal_avg['tau_ci_upper_pooled'] = df_signal_avg['tau_mean'] + 1.96 * df_signal_avg['tau_se']

    # 5. Setup Figure (wider to accommodate labels)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # 6. Plot Null Distribution (Boxplot) - pooled across progressions
    sns.boxplot(
        data=df_noise, 
        x='model', 
        y='tau', 
        ax=ax,
        color='#e0e0e0', 
        width=0.4, 
        showfliers=False,
        order=pcfg.MODEL_ORDER,
        boxprops={'linewidth': 1.5, 'edgecolor': '#555555'},
        medianprops={'color': '#333333', 'linewidth': 1.5},
        whiskerprops={'linewidth': 1.2},
        capprops={'linewidth': 1.2}
    )

    # 7. Plot True Signal (Colored Star + CI) - averaged across progressions
    for j, model in enumerate(pcfg.MODEL_ORDER):
        model_data = df_signal_avg[df_signal_avg['model'] == model]
        
        if not model_data.empty:
            val = model_data['tau_mean'].values[0]
            lower = model_data['tau_ci_lower_pooled'].values[0]
            upper = model_data['tau_ci_upper_pooled'].values[0]
            
            # Plot the Star (colored)
            ax.scatter(j, val, color=STAR_COLOR, s=250, marker='*', 
                       edgecolor='white', linewidth=0.5, zorder=10)
            
            # Plot the Error Bars
            if not np.isnan(lower) and not np.isnan(upper):
                yerr = [[val - lower], [upper - val]]
                ax.errorbar(j, val, yerr=yerr, fmt='none', ecolor='black', 
                            elinewidth=1.5, capsize=4, zorder=11)

    # 8. Formatting
    ax.set_ylabel(r"Trajectory Fidelity ($\tau$)", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax.set_xlabel("")
    
    # X-axis labels with slight rotation to prevent overlap
    labels = [pcfg.MODEL_LABELS.get(m, m) for m in pcfg.MODEL_ORDER]
    ax.set_xticklabels(labels, fontweight='bold', fontsize=pcfg.FONT_SIZES['tick_label'],
                       rotation=20, ha='right')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='grey')
    ax.xaxis.grid(False)
    
    sns.despine(ax=ax, trim=True)

    # 9. Legend (above plot, horizontal)
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=STAR_COLOR,
               markeredgecolor='white', markersize=14, label='Biological Ordering (Real)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e0e0e0', 
               markeredgecolor='#555555', markersize=10, label='Random Permutations (Null)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False,
              fontsize=pcfg.FONT_SIZES['legend'])

    # 10. Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # 11. Save
    output_subdir = OUTPUT_DIR / "permutation"
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / "permutation_plot_averaged.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")

if __name__ == "__main__":
    plot_specificity_averaged()