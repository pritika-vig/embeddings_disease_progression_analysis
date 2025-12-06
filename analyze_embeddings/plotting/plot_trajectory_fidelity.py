#!/usr/bin/env python
"""
Plotting Script: Emergence of Temporal Structure.
Compares Real Trajectory Fidelity (Kendall's Tau) vs. Null (Shuffled) Baseline.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config

# -----------------------------------------------------------------------------
# Configuration & Style
# -----------------------------------------------------------------------------

OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visual settings
COLORS = {
    "BDC": "#e74c3c",       # Red
    "CRC-Conventional": "#3498db",  # Blue
    "CRC-Serrated": "#2ecc71",  # Green
    "SCC": "#f1c40f",       # Yellow
    "Null": "#7f8c8d"       # Gray
}

OFFSETS = {
    "BDC": -0.2,
    "CRC-Conventional": -0.07,
    "CRC-Serrated": 0.07,
    "SCC": 0.2,
    # We place Nulls to the far right of the cluster
    "Null": 0.35 
}

MODEL_ORDER = config.EXPECTED_MODELS # e.g. ['uni2', 'virchow2', 'gigapath', 'conch', 'musk', 'dinov2']

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_and_prep_data():
    """Loads Real and Null CSVs and unifies columns."""
    
    # 1. Load Real Results
    try:
        df_real = pd.read_csv(config.FULL_RESULTS_OUTPUT_PATH)
        # Filter for final embedding only
        df_real = df_real[df_real["embedding_type"] == "final_embedding"].copy()
    except FileNotFoundError:
        print(f"Error: Could not find real results at {config.FULL_RESULTS_OUTPUT_PATH}")
        sys.exit(1)

    # 2. Load Null Results
    try:
        df_null = pd.read_csv(config.NULL_RESULTS_OUTPUT_PATH)
        # We treat "Null" as a distinct 'progression' for plotting purposes
        # Note: We aggregate nulls per model because the null distribution 
        # is usually similar across progressions, but if you want specific ones,
        # we can keep them split. Here we average nulls per model for cleanliness.
        df_null = df_null.groupby("model")[["tau", "tau_ci_lower", "tau_ci_upper"]].mean().reset_index()
        df_null["progression"] = "Null" 
    except FileNotFoundError:
        print(f"Warning: Could not find null results at {config.NULL_RESULTS_OUTPUT_PATH}. Plotting without nulls.")
        df_null = pd.DataFrame()

    # 3. Rename columns for easier access if necessary
    # Ensure we have consistent naming. Your previous script output: tau_ci_lower
    # but the snippet used 'lower'. Let's standardize to 'lower'/'upper'
    rename_map = {"tau_ci_lower": "lower", "tau_ci_upper": "upper"}
    df_real = df_real.rename(columns=rename_map)
    if not df_null.empty:
        df_null = df_null.rename(columns=rename_map)

    return df_real, df_null

# -----------------------------------------------------------------------------
# Main Plotting Logic
# -----------------------------------------------------------------------------

def main():
    df_real, df_null = load_and_prep_data()
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # X-Axis Base Coordinates
    x_base = np.arange(len(MODEL_ORDER))
    
    # --- 1. Plot Real Data ---
    for prog in df_real["progression"].unique():
        if prog not in COLORS: continue # Skip if not in color map
        
        subset = df_real[df_real["progression"] == prog].set_index("model")
        # Reindex to ensure we have a row for every model (fills NaNs if missing)
        subset = subset.reindex(MODEL_ORDER)
        
        # Calculate Coordinates
        xs = x_base + OFFSETS[prog]
        ys = subset["tau"].values
        
        # Calculate Relative Errors for Matplotlib
        # yerr must be shape (2, N) -> [[lower_errors], [upper_errors]]
        yerr = np.array([
            (subset["tau"] - subset["lower"]).values,
            (subset["upper"] - subset["tau"]).values
        ])
        
        # Handle NaNs (if a model failed)
        mask = ~np.isnan(ys)
        
        ax.errorbar(
            xs[mask], ys[mask], 
            yerr=yerr[:, mask],
            fmt='o', 
            label=prog, 
            color=COLORS[prog],
            ecolor=COLORS[prog],
            capsize=3, 
            elinewidth=1.5, 
            alpha=0.9,
            markersize=6
        )

    # --- 2. Plot Null Data (The Baseline) ---
    if not df_null.empty:
        subset = df_null.set_index("model").reindex(MODEL_ORDER)
        
        xs = x_base + OFFSETS["Null"]
        ys = subset["tau"].values
        
        yerr = np.array([
            (subset["tau"] - subset["lower"]).values,
            (subset["upper"] - subset["tau"]).values
        ])
        
        mask = ~np.isnan(ys)
        
        ax.errorbar(
            xs[mask], ys[mask], 
            yerr=yerr[:, mask],
            fmt='x',  # Different marker for Null
            label="Random Shuffle (Null)", 
            color=COLORS["Null"],
            ecolor=COLORS["Null"],
            capsize=3, 
            elinewidth=1.0, 
            alpha=0.7,
            markersize=6,
            linestyle='None' # No connecting lines
        )

    # --- 3. Styling & Annotation ---
    
    ax.set_xticks(x_base)
    ax.set_xticklabels(MODEL_ORDER, fontsize=11, fontweight='bold')
    ax.set_ylabel("Kendall's Tau (Trajectory Fidelity)", fontsize=12)
    ax.set_title("Emergence of Temporal Structure across Foundation Models", fontsize=14)
    
    # Legend
    ax.legend(title="Disease Progression", loc='lower left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Background Regions (Hardcoded based on your Model Order)
    # Assumes: uni2, virchow2, gigapath (Pathology) | conch, musk (V-L) | dinov2 (Natural)
    
    # Vision-Only (Pathology)
    ax.axvspan(-0.5, 2.5, color='blue', alpha=0.05, label='_nolegend_')
    ax.text(1, 0.05, "Vision-Only (Pathology)", ha='center', color='blue', alpha=0.5, fontweight='bold')

    # Vision-Language
    ax.axvspan(2.5, 4.5, color='purple', alpha=0.05, label='_nolegend_')
    ax.text(3.5, 0.05, "Vision-Language", ha='center', color='purple', alpha=0.5, fontweight='bold')

    # Natural Image
    ax.axvspan(4.5, 5.5, color='gray', alpha=0.05, label='_nolegend_')
    ax.text(5, 0.05, "Natural Image", ha='center', color='gray', alpha=0.5, fontweight='bold')

    plt.tight_layout()

    # --- 4. Save ---
    output_filename = OUTPUT_DIR / "model_comparison_with_null.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    main()