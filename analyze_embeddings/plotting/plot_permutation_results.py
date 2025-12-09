#!/usr/bin/env python
"""
Plotting Script: Stage Permutation Specificity (Final Paper Version).
Grid layout with colored stars matching Figure 1 + Global ICML Title.
"""

import sys
from pathlib import Path
import math
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
# Use the shared ICML style from your config file
pcfg.set_icml_style()

PERM_FILE = config.PERMUTATION_RESULTS_OUTPUT_PATH
REAL_FILE = config.FULL_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The specific embedding type that contains the canonical signal + CIs
TARGET_EMBEDDING = 'final_embedding'

def plot_specificity_grid():
    # 1. Load Data
    df_real = pd.read_csv(REAL_FILE)
    df_perm = pd.read_csv(PERM_FILE)

    # 2. Filter Data
    df_noise = df_perm[df_perm['type'] == 'permuted'].copy()
    df_signal = df_real[df_real['embedding_type'] == TARGET_EMBEDDING].copy()

    # --- Enforce Shared Model Order ---
    df_noise['model'] = pd.Categorical(df_noise['model'], categories=pcfg.MODEL_ORDER, ordered=True)
    df_signal['model'] = pd.Categorical(df_signal['model'], categories=pcfg.MODEL_ORDER, ordered=True)
    
    # Sort data
    df_noise = df_noise.sort_values('model')
    df_signal = df_signal.sort_values('model')

    unique_progressions = df_noise['progression'].unique()
    num_progs = len(unique_progressions)
    
    # 3. Setup Grid (2 Columns)
    cols = 2
    rows = math.ceil(num_progs / cols)
    
    # Slightly taller figure to accommodate the new Global Title
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.0 * rows), 
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, prog in enumerate(unique_progressions):
        ax = axes_flat[i]
        
        # --- LOOKUP COLOR FROM CONFIG ---
        star_color = pcfg.PROGRESSION_COLORS.get(prog, '#D62728')
        
        # Subset data
        noise_subset = df_noise[df_noise['progression'] == prog]
        signal_subset = df_signal[df_signal['progression'] == prog]
        
        # --- A. Plot Null Distribution (Boxplot) ---
        sns.boxplot(data=noise_subset, x='model', y='tau', ax=ax,
                    color='#e0e0e0', width=0.5, showfliers=False,
                    boxprops={'linewidth': 1.5, 'edgecolor': '#555555'},
                    medianprops={'color': '#333333', 'linewidth': 1.5})
        
        # Strip plot (Raw points)
        # sns.stripplot(data=noise_subset, x='model', y='tau', ax=ax,
        #               color='#333333', alpha=0.3, jitter=0.2, size=3, zorder=0)
        
        # --- B. Plot True Signal (Star + CI) ---
        for j, model in enumerate(pcfg.MODEL_ORDER):
            model_signal = signal_subset[signal_subset['model'] == model]
            
            if not model_signal.empty:
                val = model_signal['tau'].values[0]
                lower = model_signal['tau_ci_lower'].values[0]
                upper = model_signal['tau_ci_upper'].values[0]
                
                # 1. Plot the Star (Colored by Disease)
                ax.scatter(j, val, color=star_color, s=250, marker='*', 
                           edgecolor='white', linewidth=0.5, zorder=10)
                
                # 2. Plot the Error Bars
                if not np.isnan(lower) and not np.isnan(upper):
                    yerr = [[val - lower], [upper - val]]
                    ax.errorbar(j, val, yerr=yerr, fmt='none', ecolor='black', 
                                elinewidth=1.8, capsize=5, zorder=11)

        # --- Formatting ---
        ax.set_title(f"{prog}", fontweight='bold', fontsize=14, pad=10)
        
        if i % cols == 0:
            ax.set_ylabel(r"Trajectory Fidelity ($\tau$)", labelpad=10)
        else:
            ax.set_ylabel("")
            
        if i >= (rows - 1) * cols:
            labels = [pcfg.MODEL_LABELS.get(m, m) for m in pcfg.MODEL_ORDER]
            ax.set_xticklabels(labels, rotation=0, fontweight='bold', fontsize=10)
            ax.set_xlabel("")
        else:
            ax.set_xlabel("")

        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='grey')
        ax.xaxis.grid(False)
        sns.despine(ax=ax, trim=True)

    # Hide unused axes
    for k in range(i + 1, len(axes_flat)):
        axes_flat[k].axis('off')

    # --- GLOBAL TITLE & LEGEND ---
    
    # 1. Add Global Title
    fig.suptitle("Specificity of trajectory fidelity: Permutation testing against null stage orders", 
                 fontsize=16, fontweight='bold', y=0.98)

    # 2. Add Legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', 
               markersize=14, label='Biological Ordering (Real)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0e0e0', 
               markeredgecolor='grey', markersize=8, label='Random Permutations (Null)')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=12)

    # 3. Adjust Layout
    # rect=[left, bottom, right, top] -> top=0.90 reserves top 10% for title/legend
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    plt.savefig(OUTPUT_DIR / 'specificity_plot_final.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_DIR / 'specificity_plot_final.png'}")

if __name__ == "__main__":
    plot_specificity_grid()