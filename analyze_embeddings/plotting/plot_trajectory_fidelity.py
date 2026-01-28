#!/usr/bin/env python
"""
Plotting Script: Emergence of Temporal Structure.
Compares Real Trajectory Fidelity (Kendall's Tau) vs. Null (Shuffled) Baseline.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# Import Root Config
sys.path.append(str(Path(__file__).resolve().parent.parent))
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
    "SCC": 0.2,
    "Null": 0.35 
}

def load_and_prep_data():
    """Loads Real and Null CSVs and unifies columns."""
    try:
        df_real = pd.read_csv(config.FULL_RESULTS_OUTPUT_PATH)
        df_real = df_real[df_real["embedding_type"] == "final_embedding"].copy()
    except FileNotFoundError:
        print(f"Error: Could not find real results at {config.FULL_RESULTS_OUTPUT_PATH}")
        sys.exit(1)

    try:
        df_null = pd.read_csv(config.NULL_RESULTS_OUTPUT_PATH)
        df_null = df_null.groupby("model")[["tau", "tau_ci_lower", "tau_ci_upper"]].mean().reset_index()
        df_null["progression"] = "Null" 
    except FileNotFoundError:
        df_null = pd.DataFrame()

    rename_map = {"tau_ci_lower": "lower", "tau_ci_upper": "upper"}
    df_real = df_real.rename(columns=rename_map)
    if not df_null.empty:
        df_null = df_null.rename(columns=rename_map)

    return df_real, df_null

def main():
    df_real, df_null = load_and_prep_data()
    
    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(15, 6)) 
    
    # Use Standardized Model Order
    x_base = np.arange(len(pcfg.MODEL_ORDER))
    
    # --- 2. Plot Real Data ---
    for prog in df_real["progression"].unique():
        if prog not in pcfg.PROGRESSION_COLORS: continue
        
        subset = df_real[df_real["progression"] == prog].set_index("model")
        subset = subset.reindex(pcfg.MODEL_ORDER) # Enforce Order
        
        xs = x_base + OFFSETS[prog]
        ys = subset["tau"].values
        
        yerr = np.array([
            (subset["tau"] - subset["lower"]).values,
            (subset["upper"] - subset["tau"]).values
        ])
        
        mask = ~np.isnan(ys)
        
        ax.errorbar(
            xs[mask], ys[mask], yerr=yerr[:, mask],
            fmt='o', 
            color=pcfg.PROGRESSION_COLORS[prog],
            ecolor=pcfg.PROGRESSION_COLORS[prog],
            capsize=3, elinewidth=1.5, alpha=0.9, markersize=7 
        )

    # --- 3. Plot Null Data ---
    if not df_null.empty:
        subset = df_null.set_index("model").reindex(pcfg.MODEL_ORDER)
        xs = x_base + OFFSETS["Null"]
        ys = subset["tau"].values
        yerr = np.array([
            (subset["tau"] - subset["lower"]).values,
            (subset["upper"] - subset["tau"]).values
        ])
        mask = ~np.isnan(ys)
        
        ax.errorbar(
            xs[mask], ys[mask], yerr=yerr[:, mask],
            fmt='x', 
            color=pcfg.PROGRESSION_COLORS["Null"],
            ecolor=pcfg.PROGRESSION_COLORS["Null"],
            capsize=3, elinewidth=1.5, alpha=0.9, markersize=8,
            markeredgewidth=2,
            linestyle='None'
        )

    # --- 4. Styling & Annotation ---
    xtick_labels = [pcfg.MODEL_LABELS.get(m, m) for m in pcfg.MODEL_ORDER]
    
    ax.set_xticks(x_base)
    ax.set_xticklabels(xtick_labels, fontweight='bold', fontsize=pcfg.FONT_SIZES['tick_label'])
    ax.set_ylabel("Kendall's Tau (Trajectory Fidelity)", fontsize=pcfg.FONT_SIZES['axis_label'])
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Clean Y-Axis limits to ensure Nulls aren't cut off
    ax.set_ylim(bottom=-0.1, top=0.90)
    ax.set_yticks(np.arange(0.0, 0.9, 0.1))

    # --- Background Regions & Category Labels ---
    # We use a transform to place labels relative to the X-Axis line
    trans = ax.get_xaxis_transform()
    label_y_pos = -0.14 # Push text down to clear the model names
    category_fontsize = pcfg.FONT_SIZES.get('category_label', 12)
    
    # 1. Natural Image
    ax.axvspan(-0.5, 0.5, color='gray', alpha=0.08, lw=0)
    ax.text(0, label_y_pos, "Natural Image", ha='center', va='top',
            color=pcfg.CATEGORY_LABEL_COLORS['natural'], 
            fontweight='bold', fontsize=category_fontsize, transform=trans)

    # 2. Vision-Language
    ax.axvspan(0.5, 2.5, color='purple', alpha=0.08, lw=0)
    ax.text(1.5, label_y_pos, "Vision-Language (Pathology)", ha='center', va='top',
            color=pcfg.CATEGORY_LABEL_COLORS['vision_language'], 
            fontweight='bold', fontsize=category_fontsize, transform=trans)

    # 3. Pathology
    ax.axvspan(2.5, 5.5, color='blue', alpha=0.08, lw=0)
    ax.text(4.0, label_y_pos, "Vision-Only (Pathology)", ha='center', va='top',
            color=pcfg.CATEGORY_LABEL_COLORS['vision_only'], 
            fontweight='bold', fontsize=category_fontsize, transform=trans)

    # --- 5. Global Legend & Title (The "Header Zone" Fix) ---
    
    # Global Title
    # fig.suptitle("Trajectory fidelity across 6 foundation models and 4 disease progressions", 
    #              fontsize=pcfg.FONT_SIZES['plot_title'], fontweight='bold', y=0.98)

    # Create Legend Handles
    legend_handles = []
    for prog, color in pcfg.PROGRESSION_COLORS.items():
        if prog == "Null": continue
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                   markersize=10, label=prog)  # Larger legend markers
        )
    legend_handles.append(
        Line2D([0], [0], marker='x', color=pcfg.PROGRESSION_COLORS["Null"], 
               linestyle='None', markeredgewidth=2.5, markersize=9, label='Random Shuffle (Null)')
    )

    # Place Legend (High up, horizontally)
    fig.legend(handles=legend_handles, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.93), 
               ncol=5, 
               frameon=False,
               fontsize=pcfg.FONT_SIZES['legend'])

    # Force the plot area down to 88% height to make room for Title/Legend
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    output_subdir = OUTPUT_DIR / "trajectory_fidelity"
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / f"model_comparison_with_null.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved to {output_path}")

if __name__ == "__main__":
    main()
# #!/usr/bin/env python
# """
# Plotting Script: Emergence of Temporal Structure.
# Compares Real Trajectory Fidelity (Kendall's Tau) vs. Null (Shuffled) Baseline.
# """

# import sys
# from pathlib import Path
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.lines import Line2D

# # Import Root Config
# sys.path.append(str(Path(__file__).resolve().parent.parent))
# import config

# # Import Local Plotting Config
# import plotting.plot_config as pcfg

# # -----------------------------------------------------------------------------
# # Configuration
# # -----------------------------------------------------------------------------
# pcfg.set_icml_style() # Apply Shared Style

# OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Offsets for grouping bars around the tick mark
# OFFSETS = {
#     "BDC": -0.2,
#     "CRC-Conventional": -0.07,
#     "CRC-Serrated": 0.07,
#     "SCC": 0.2,
#     "Null": 0.35 
# }

# def load_and_prep_data():
#     """Loads Real and Null CSVs and unifies columns."""
#     try:
#         df_real = pd.read_csv(config.FULL_RESULTS_OUTPUT_PATH)
#         df_real = df_real[df_real["embedding_type"] == "final_embedding"].copy()
#     except FileNotFoundError:
#         print(f"Error: Could not find real results at {config.FULL_RESULTS_OUTPUT_PATH}")
#         sys.exit(1)

#     try:
#         df_null = pd.read_csv(config.NULL_RESULTS_OUTPUT_PATH)
#         df_null = df_null.groupby("model")[["tau", "tau_ci_lower", "tau_ci_upper"]].mean().reset_index()
#         df_null["progression"] = "Null" 
#     except FileNotFoundError:
#         df_null = pd.DataFrame()

#     rename_map = {"tau_ci_lower": "lower", "tau_ci_upper": "upper"}
#     df_real = df_real.rename(columns=rename_map)
#     if not df_null.empty:
#         df_null = df_null.rename(columns=rename_map)

#     return df_real, df_null

# def main():
#     df_real, df_null = load_and_prep_data()
    
#     # 1. Setup Figure
#     fig, ax = plt.subplots(figsize=(15, 6)) 
    
#     # Use Standardized Model Order
#     x_base = np.arange(len(pcfg.MODEL_ORDER))
    
#     # --- 2. Plot Real Data ---
#     for prog in df_real["progression"].unique():
#         if prog not in pcfg.PROGRESSION_COLORS: continue
        
#         subset = df_real[df_real["progression"] == prog].set_index("model")
#         subset = subset.reindex(pcfg.MODEL_ORDER) # Enforce Order
        
#         xs = x_base + OFFSETS[prog]
#         ys = subset["tau"].values
        
#         yerr = np.array([
#             (subset["tau"] - subset["lower"]).values,
#             (subset["upper"] - subset["tau"]).values
#         ])
        
#         mask = ~np.isnan(ys)
        
#         ax.errorbar(
#             xs[mask], ys[mask], yerr=yerr[:, mask],
#             fmt='o', 
#             color=pcfg.PROGRESSION_COLORS[prog],
#             ecolor=pcfg.PROGRESSION_COLORS[prog],
#             capsize=3, elinewidth=1.5, alpha=0.9, markersize=6
#         )

#     # --- 3. Plot Null Data ---
#     if not df_null.empty:
#         subset = df_null.set_index("model").reindex(pcfg.MODEL_ORDER)
#         xs = x_base + OFFSETS["Null"]
#         ys = subset["tau"].values
#         yerr = np.array([
#             (subset["tau"] - subset["lower"]).values,
#             (subset["upper"] - subset["tau"]).values
#         ])
#         mask = ~np.isnan(ys)
        
#         ax.errorbar(
#             xs[mask], ys[mask], yerr=yerr[:, mask],
#             fmt='x', 
#             color=pcfg.PROGRESSION_COLORS["Null"],
#             ecolor=pcfg.PROGRESSION_COLORS["Null"],
#             capsize=3, elinewidth=1.0, alpha=0.7, markersize=6, linestyle='None'
#         )

#     # --- 4. Styling & Annotation ---
#     xtick_labels = [pcfg.MODEL_LABELS.get(m, m) for m in pcfg.MODEL_ORDER]
    
#     ax.set_xticks(x_base)
#     ax.set_xticklabels(xtick_labels, fontweight='bold', fontsize=11)
#     ax.set_ylabel("Kendall's Tau (Trajectory Fidelity)", fontsize=12)
    
#     ax.grid(True, axis='y', linestyle='--', alpha=0.3)

#     # Clean Y-Axis limits to ensure Nulls aren't cut off
#     ax.set_ylim(bottom=-0.1, top=0.90)
#     ax.set_yticks(np.arange(0.0, 0.9, 0.1))

#     # --- Background Regions & Category Labels ---
#     # We use a transform to place labels relative to the X-Axis line
#     trans = ax.get_xaxis_transform()
#     label_y_pos = -0.14 # Push text down to clear the model names
    
#     # 1. Natural Image
#     ax.axvspan(-0.5, 0.5, color='gray', alpha=0.05, lw=0)
#     ax.text(0, label_y_pos, "Natural Image", ha='center', va='top',
#             color='gray', alpha=0.6, fontweight='bold', transform=trans)

#     # 2. Vision-Language
#     ax.axvspan(0.5, 2.5, color='purple', alpha=0.05, lw=0)
#     ax.text(1.5, label_y_pos, "Vision-Language (Pathology)", ha='center', va='top',
#             color='purple', alpha=0.6, fontweight='bold', transform=trans)

#     # 3. Pathology
#     ax.axvspan(2.5, 5.5, color='blue', alpha=0.05, lw=0)
#     ax.text(4.0, label_y_pos, "Vision-Only (Pathology)", ha='center', va='top',
#             color='blue', alpha=0.6, fontweight='bold', transform=trans)

#     # --- 5. Global Legend & Title (The "Header Zone" Fix) ---
    
#     # Global Title
#     fig.suptitle("Trajectory fidelity across 6 foundation models and 4 disease progressions", 
#                  fontsize=14, fontweight='bold', y=0.98)

#     # Create Legend Handles
#     legend_handles = []
#     for prog, color in pcfg.PROGRESSION_COLORS.items():
#         if prog == "Null": continue
#         legend_handles.append(
#             Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
#                    markersize=9, label=prog)
#         )
#     legend_handles.append(
#         Line2D([0], [0], marker='x', color=pcfg.PROGRESSION_COLORS["Null"], 
#                linestyle='None', markeredgewidth=2, markersize=8, label='Random Shuffle (Null)')
#     )

#     # Place Legend (High up, horizontally)
#     fig.legend(handles=legend_handles, 
#                loc='upper center', 
#                bbox_to_anchor=(0.5, 0.93), 
#                ncol=5, 
#                frameon=False,
#                fontsize=10)

#     # Force the plot area down to 88% height to make room for Title/Legend
#     plt.tight_layout(rect=[0, 0, 1, 0.88])

#     output_subdir = OUTPUT_DIR / "trajectory_fidelity"
#     output_subdir.mkdir(parents=True, exist_ok=True)
#     output_path = output_subdir / f"model_comparison_with_null.png"
#     #output_filename = OUTPUT_DIR / "model_comparison_with_null.png"
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"✅ Plot successfully saved to {output_path}")

# if __name__ == "__main__":
#     main()