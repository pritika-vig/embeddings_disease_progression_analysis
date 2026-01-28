#!/usr/bin/env python
"""
Paper Figure: Biological Validation (Panel B - Final v2).

Changes:
- Removed text labels from inside the plot.
- Added a full-text "Stage Legend" below the X-axis.
- Maintained the "box" visualization for distributions.

Usage:
    python plotting/figure_biological_validation_final_v2.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent 
results_dir = project_root / "results"
plots_dir = project_root / "plots" / "biological_validation"
plots_dir.mkdir(parents=True, exist_ok=True)

COLORS = {
    'Cancer Cells': '#D62828',       # Red
    'Fibroblasts': '#8D6E63',        # Brown
    'Apoptotic Bodies': '#9C27B0',   # Purple
    'Neutrophils': '#F4A261',        # Orange
}

STAGE_COLORS = {
    'Adenoma low grade': '#2ECC71',      # Green
    'Adenoma high grade': '#F1C40F',     # Yellow
    'Adenocarcinoma low grade': '#E67E22', # Orange
    'Adenocarcinoma high grade': '#C0392B' # Red
}

STAGE_ORDER = [
    'Adenoma low grade', 
    'Adenoma high grade', 
    'Adenocarcinoma low grade', 
    'Adenocarcinoma high grade'
]

def load_data(filepath):
    if not filepath.exists():
        print(f"❌ Error: File not found at {filepath}")
        sys.exit(1)
    df = pd.read_csv(filepath)
    return df.sort_values('pseudotime_norm').reset_index(drop=True)

def get_smoothed_data(df, columns, window_fraction=0.15):
    window = int(len(df) * window_fraction)
    smoothed = df[columns].rolling(window=window, center=True, min_periods=window//4).mean()
    smoothed['pseudotime'] = df['pseudotime_norm']
    return smoothed.dropna()

def plot_panel_b_final(df, output_path):
    print("🎨 Generating Panel B (Final v2)...")
    
    # Increase height slightly to accommodate the bottom legend
    fig, ax = plt.subplots(figsize=(5, 4.8), dpi=300)
    
    # 1. Plot Z-Score Lines
    z_features = {
        'cancer_cells_prop': ('Cancer Cells', COLORS['Cancer Cells'], '--'),
        'fibroblasts_prop': ('Fibroblasts', COLORS['Fibroblasts'], '-'),
        'apoptotic_bodies_prop': ('Apoptosis', COLORS['Apoptotic Bodies'], '-'),
        'neutrophils_prop': ('Neutrophils', COLORS['Neutrophils'], '-')
    }
    
    cols = list(z_features.keys())
    smooth_z = get_smoothed_data(df, cols)
    
    # Calculate Z-Scores
    z_df = (smooth_z[cols] - smooth_z[cols].mean()) / smooth_z[cols].std()
    z_df['pseudotime'] = smooth_z['pseudotime']
    
    for col, (label, color, style) in z_features.items():
        alpha = 0.5 if 'cancer' in col else 1.0
        width = 2.0
        ax.plot(z_df['pseudotime'], z_df[col], label=label, 
                color=color, linestyle=style, linewidth=width, alpha=alpha)

    # 2. Add Stage Boxes (No Text Labels)
    y_min, y_max = -2.5, 4.0
    box_height = 0.25
    box_y_pos = y_min + 0.1
    
    for stage in STAGE_ORDER:
        mask = df['class_label'] == stage
        if not mask.any(): continue
        
        pts = df.loc[mask, 'pseudotime_norm']
        q25, q75 = pts.quantile(0.25), pts.quantile(0.75)
        median = pts.median()
        color = STAGE_COLORS.get(stage, 'gray')
        
        # Box
        rect = mpatches.Rectangle((q25, box_y_pos), q75 - q25, box_height, 
                                  facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        # Median Line
        ax.plot([median, median], [box_y_pos, box_y_pos + box_height], 
                color='black', linewidth=1.5)

    # 3. Formatting
    ax.set_ylabel('Standardized Abundance (Z-Score)', fontsize=10)
    ax.set_xlabel('Diffusion Pseudotime', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    
    # Feature Legend (Top Left)
    legend1 = ax.legend(loc='upper left', ncol=2, frameon=True, fontsize=8, 
                        columnspacing=1.0, handlelength=1.5)
    ax.add_artist(legend1) # Keep this legend when adding the next one

    # 4. Create "Stage Key" Below X-Axis
    stage_patches = [mpatches.Patch(facecolor=STAGE_COLORS[s], edgecolor='black', 
                                    alpha=0.6, label=s) for s in STAGE_ORDER]
    
    # Place this legend below the axis title
    fig.legend(handles=stage_patches, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=2, 
               fontsize=8, frameon=False, title="")

    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout to make room for the bottom legend
    plt.subplots_adjust(bottom=0.22)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

# =============================================================================
# APPENDIX DIAGNOSTICS (drop-in)
# =============================================================================

def _ensure_appendix_dir(base_dir: Path) -> Path:
    appendix_dir = base_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    return appendix_dir


def plot_pseudotime_density(df: pd.DataFrame, out_dir: Path, bins: int = 60):
    """
    A) Global pseudotime density overall + per class.
    Saves:
      - pseudotime_density_overall.png
      - pseudotime_density_by_stage.png
    """
    appendix_dir = _ensure_appendix_dir(out_dir)

    # Overall histogram
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)
    ax.hist(df["pseudotime_norm"].values, bins=bins)
    ax.set_xlabel("Diffusion Pseudotime")
    ax.set_ylabel("Count")
    ax.set_title("Pseudotime density (all patches)")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(appendix_dir / "pseudotime_density_overall.png", bbox_inches="tight")
    plt.close(fig)

    # Per-stage hist overlays (4 series)
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)
    for stage in STAGE_ORDER:
        mask = df["class_label"] == stage
        if not mask.any():
            continue
        ax.hist(
            df.loc[mask, "pseudotime_norm"].values,
            bins=bins,
            alpha=0.5,
            label=stage,
        )
    ax.set_xlabel("Diffusion Pseudotime")
    ax.set_ylabel("Count")
    ax.set_title("Pseudotime density by stage (hist overlays)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(appendix_dir / "pseudotime_density_by_stage.png", bbox_inches="tight")
    plt.close(fig)


def plot_local_effective_sample_size(
    df: pd.DataFrame,
    out_dir: Path,
    window_fraction: float = 0.15,
    bins: int = 80,
):
    """
    B) Local support: counts per pseudotime bin + "effective N" in a rolling window.

    This aligns with your smoother, which uses a rolling window of:
        window = int(len(df) * window_fraction)

    Saves:
      - pseudotime_local_counts_and_windowN.png
    """
    appendix_dir = _ensure_appendix_dir(out_dir)

    pts = df["pseudotime_norm"].values
    N = len(df)
    window = max(5, int(N * window_fraction))

    # Bin counts
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_counts, _ = np.histogram(pts, bins=edges)

    # Rolling-window "effective N" at each bin center:
    # count points within [center - w/2, center + w/2] in pseudotime.
    # We approximate with a fixed pseudotime half-width set by the empirical window size.
    pts_sorted = np.sort(pts)
    # Approximate pseudotime half-width using median spacing * window/2
    # (robust to uneven density, but still only an approximation).
    if len(pts_sorted) >= 2:
        median_dx = np.median(np.diff(pts_sorted))
    else:
        median_dx = 1.0
    half_width = (window / 2.0) * median_dx

    window_counts = []
    for c in bin_centers:
        lo = c - half_width
        hi = c + half_width
        # count in [lo, hi]
        left = np.searchsorted(pts_sorted, lo, side="left")
        right = np.searchsorted(pts_sorted, hi, side="right")
        window_counts.append(right - left)
    window_counts = np.asarray(window_counts)

    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=300)
    ax.plot(bin_centers, bin_counts, label=f"Bin counts (bins={bins})")
    ax.plot(bin_centers, window_counts, label=f"Approx. window support (N~{window})")
    ax.set_xlabel("Diffusion Pseudotime")
    ax.set_ylabel("Count")
    ax.set_title("Local sampling support over pseudotime")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        appendix_dir / "pseudotime_local_counts_and_windowN.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_smoothing_robustness(
    df: pd.DataFrame,
    out_dir: Path,
    columns=None,
    window_fractions=(0.10, 0.15, 0.20),
):
    """
    C) Robustness to smoothing: overlay smoothed Z-score curves under multiple window fractions.

    Saves:
      - smoothing_robustness_fibroblasts.png
      - smoothing_robustness_all_features.png
    """
    appendix_dir = _ensure_appendix_dir(out_dir)

    if columns is None:
        columns = [
            "cancer_cells_prop",
            "fibroblasts_prop",
            "apoptotic_bodies_prop",
            "neutrophils_prop",
        ]

    # --- (1) Fibroblast-only robustness (most relevant to "V-shape" claims) ---
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)
    for wf in window_fractions:
        smooth = get_smoothed_data(df, ["fibroblasts_prop"], window_fraction=wf)
        z = (smooth["fibroblasts_prop"] - smooth["fibroblasts_prop"].mean()) / smooth[
            "fibroblasts_prop"
        ].std()
        ax.plot(smooth["pseudotime"], z.values, label=f"window_fraction={wf:.2f}")
    ax.set_xlabel("Diffusion Pseudotime")
    ax.set_ylabel("Fibroblasts (Z-score)")
    ax.set_title("Smoothing robustness (fibroblasts)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(appendix_dir / "smoothing_robustness_fibroblasts.png", bbox_inches="tight")
    plt.close(fig)

    # --- (2) All-feature robustness (small multiples in one panel is tempting,
    # but keeping it readable: overlay each feature separately in the same axis via linestyle)
    # If you prefer one-feature-per-plot, tell me and I'll adjust.
    feature_specs = {
        "cancer_cells_prop": ("Cancer Cells", COLORS.get("Cancer Cells", None), "--"),
        "fibroblasts_prop": ("Fibroblasts", COLORS.get("Fibroblasts", None), "-"),
        "apoptotic_bodies_prop": ("Apoptosis", COLORS.get("Apoptotic Bodies", None), "-"),
        "neutrophils_prop": ("Neutrophils", COLORS.get("Neutrophils", None), "-"),
    }

    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=300)
    for wf in window_fractions:
        smooth = get_smoothed_data(df, columns, window_fraction=wf)
        z = (smooth[columns] - smooth[columns].mean()) / smooth[columns].std()
        for col, (label, color, style) in feature_specs.items():
            if col not in columns:
                continue
            alpha = 0.35 if "cancer" in col else 0.6
            ax.plot(
                smooth["pseudotime"],
                z[col].values,
                linestyle=style,
                alpha=alpha,
                label=f"{label} (wf={wf:.2f})",
            )
    ax.set_xlabel("Diffusion Pseudotime")
    ax.set_ylabel("Z-score")
    ax.set_title("Smoothing robustness (all features)")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Make legend compact
    ax.legend(
        frameon=False,
        fontsize=6,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.4,
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(appendix_dir / "smoothing_robustness_all_features.png", bbox_inches="tight")
    plt.close(fig)


def save_fibroblast_binned_bootstrap_figure(
    df: pd.DataFrame,
    out_dir: Path,
    n_bins: int = 25,
    n_boot: int = 500,
    seed: int = 0,
    min_n_visible: int = 50,
):
    """
    Create a binned fibroblast vs pseudotime figure with bootstrap CIs.
    Saves a PNG into <out_dir>/appendix/.

    Figure shows:
      - mean fibroblast proportion per bin
      - 90% bootstrap CI (shaded)
      - bins with low support (n < min_n_visible) faded
    """
    appendix_dir = _ensure_appendix_dir(out_dir)

    d = df[["pseudotime_norm", "fibroblasts_prop"]].dropna().copy()
    d = d.sort_values("pseudotime_norm").reset_index(drop=True)

    # Bin pseudotime
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.digitize(d["pseudotime_norm"].values, edges, right=False) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    rng = np.random.default_rng(seed)

    means, ci5s, ci95s, ns = [], [], [], []

    for b in range(n_bins):
        mask = bin_id == b
        y = d.loc[mask, "fibroblasts_prop"].values
        n = int(mask.sum())
        ns.append(n)

        if n == 0:
            means.append(np.nan)
            ci5s.append(np.nan)
            ci95s.append(np.nan)
            continue

        mean = np.mean(y)
        means.append(mean)

        if n_boot > 0 and n > 1:
            boot_means = np.empty(n_boot)
            for i in range(n_boot):
                samp = rng.choice(y, size=n, replace=True)
                boot_means[i] = np.mean(samp)
            ci5, ci95 = np.quantile(boot_means, [0.05, 0.95])
        else:
            ci5, ci95 = np.nan, np.nan

        ci5s.append(ci5)
        ci95s.append(ci95)

    means = np.asarray(means)
    ci5s = np.asarray(ci5s)
    ci95s = np.asarray(ci95s)
    ns = np.asarray(ns)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)

    # CI band
    ax.fill_between(
        centers,
        ci5s,
        ci95s,
        color="#8D6E63",  # fibroblast brown
        alpha=0.25,
        linewidth=0,
        label="90% bootstrap CI",
    )

    # Mean line
    ax.plot(
        centers,
        means,
        color="#8D6E63",
        linewidth=2.0,
        label="Binned mean",
    )

    # Fade low-support bins
    low_support = ns < min_n_visible
    ax.scatter(
        centers[low_support],
        means[low_support],
        color="#8D6E63",
        alpha=0.35,
        s=20,
        zorder=3,
    )

    ax.set_xlabel("Diffusion Pseudotime")
    ax.set_ylabel("Fibroblast Proportion")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        appendix_dir / "fibroblasts_binned_bootstrap.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def save_stage_summary_csv(df: pd.DataFrame, out_dir: Path):
    """
    (2) Stage-level summary CSV for appendix:
        - n_patches
        - pseudotime median/IQR
        - fibroblasts_prop median/IQR
        - fibroblasts_prop mean/std

    Output in <out_dir>/appendix/:
      - stage_summary_fibroblasts.csv
    """
    appendix_dir = _ensure_appendix_dir(out_dir)

    d = df[["class_label", "pseudotime_norm", "fibroblasts_prop"]].dropna().copy()

    def iqr(x):
        q25, q75 = np.quantile(x, [0.25, 0.75])
        return float(q25), float(q75)

    rows = []
    for stage in STAGE_ORDER:
        s = d[d["class_label"] == stage]
        if len(s) == 0:
            continue

        pt = s["pseudotime_norm"].values
        fb = s["fibroblasts_prop"].values

        pt_q25, pt_q75 = iqr(pt)
        fb_q25, fb_q75 = iqr(fb)

        rows.append(
            {
                "stage": stage,
                "n_patches": int(len(s)),
                "pseudotime_median": float(np.median(pt)),
                "pseudotime_q25": pt_q25,
                "pseudotime_q75": pt_q75,
                "fibroblasts_prop_median": float(np.median(fb)),
                "fibroblasts_prop_q25": fb_q25,
                "fibroblasts_prop_q75": fb_q75,
                "fibroblasts_prop_mean": float(np.mean(fb)),
                "fibroblasts_prop_std": float(np.std(fb, ddof=1)) if len(fb) > 1 else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out_path = appendix_dir / "stage_summary_fibroblasts.csv"
    out.to_csv(out_path, index=False)


def plot_cell_type_zscore_and_raw(
    df: pd.DataFrame,
    out_dir: Path,
    window_fraction: float = 0.15,
):
    """
    Three-panel figure for appendix:
      - Top panel: Raw cell proportions for all cell types
      - Middle panel: Z-score standardized abundance for all cell types  
      - Bottom panel: Stage distribution boxes along pseudotime (IQR + median)
    
    This provides the full detail behind the main figure's simplified view,
    with stage context to help interpret where trends occur in disease progression.
    
    Saves:
      - cell_types_zscore_and_raw.png
    """
    appendix_dir = _ensure_appendix_dir(out_dir)
    
    # All 13 cell types with colors (all solid lines)
    ALL_CELL_TYPES = {
        'cancer_cells_prop': ('Cancer Cells', '#E63946', '-'),
        'epithelial_normal_prop': ('Normal Epithelium', '#06D6A0', '-'),
        'fibroblasts_prop': ('Fibroblasts', '#8D6E63', '-'),
        'lymphocytes_prop': ('Lymphocytes', '#457B9D', '-'),
        'macrophages_prop': ('Macrophages', '#9D4EDD', '-'),
        'plasmocytes_prop': ('Plasmocytes', '#E91E63', '-'),
        'neutrophils_prop': ('Neutrophils', '#F4A261', '-'),
        'eosinophils_prop': ('Eosinophils', '#FFB300', '-'),
        'endothelial_prop': ('Endothelial', '#00BCD4', '-'),
        'apoptotic_bodies_prop': ('Apoptotic Bodies', '#607D8B', '-'),
        'smooth_muscle_prop': ('Smooth Muscle', '#795548', '-'),
        'red_blood_cells_prop': ('Red Blood Cells', '#D32F2F', '-'),
        'mitotic_figures_prop': ('Mitotic Figures', '#7B1FA2', '-'),
    }
    
    # Filter to columns that exist in the dataframe
    feature_specs = {col: spec for col, spec in ALL_CELL_TYPES.items() if col in df.columns}
    
    if not feature_specs:
        print("⚠️ No cell type columns found in dataframe")
        return
    
    cols = list(feature_specs.keys())
    smooth = get_smoothed_data(df, cols, window_fraction=window_fraction)
    
    # Calculate Z-scores
    z_df = (smooth[cols] - smooth[cols].mean()) / smooth[cols].std()
    z_df['pseudotime'] = smooth['pseudotime']
    
    # Create figure with 3 panels - raw on top, z-score middle, stages bottom
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), dpi=300, 
                              gridspec_kw={'height_ratios': [3, 3, 1.2], 'hspace': 0.15})
    ax_raw, ax_zscore, ax_stages = axes
    
    # --- Top Panel: Raw Proportions ---
    for col, (label, color, style) in feature_specs.items():
        ax_raw.plot(
            smooth['pseudotime'], 
            smooth[col], 
            label=label,
            color=color, 
            linestyle=style, 
            linewidth=1.5, 
            alpha=0.85
        )
    
    ax_raw.set_ylabel('Cell Proportion', fontsize=10)
    ax_raw.set_xlim(0, 1)
    ax_raw.legend(loc='upper right', ncol=3, frameon=True, fontsize=7,
                  columnspacing=0.8, handlelength=1.5)
    ax_raw.grid(True, alpha=0.15)
    ax_raw.spines['top'].set_visible(False)
    ax_raw.spines['right'].set_visible(False)
    ax_raw.set_title('Cell Type Proportions Along Pseudotime', fontsize=11, fontweight='bold', loc='left')
    plt.setp(ax_raw.get_xticklabels(), visible=False)
    
    # --- Middle Panel: Z-Scores ---
    for col, (label, color, style) in feature_specs.items():
        ax_zscore.plot(
            z_df['pseudotime'], 
            z_df[col], 
            label=label,
            color=color, 
            linestyle=style, 
            linewidth=1.5, 
            alpha=0.85
        )
    
    ax_zscore.set_ylabel('Standardized Abundance\n(Z-Score)', fontsize=10)
    ax_zscore.set_xlim(0, 1)
    ax_zscore.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax_zscore.legend(loc='upper right', ncol=3, frameon=True, fontsize=7, 
                     columnspacing=0.8, handlelength=1.5)
    ax_zscore.grid(True, alpha=0.15)
    ax_zscore.spines['top'].set_visible(False)
    ax_zscore.spines['right'].set_visible(False)
    ax_zscore.set_title('Standardized Cell Type Abundance', fontsize=11, fontweight='bold', loc='left')
    plt.setp(ax_zscore.get_xticklabels(), visible=False)
    
    # --- Bottom Panel: Stage Distribution Boxes ---
    # Draw horizontal box-and-whisker style representation for each stage
    box_height = 0.6
    y_positions = np.arange(len(STAGE_ORDER))
    
    for i, stage in enumerate(STAGE_ORDER):
        mask = df['class_label'] == stage
        if not mask.any():
            continue
        
        pts = df.loc[mask, 'pseudotime_norm']
        q25, q75 = pts.quantile(0.25), pts.quantile(0.75)
        median = pts.median()
        min_val, max_val = pts.min(), pts.max()
        color = STAGE_COLORS.get(stage, 'gray')
        
        # Whiskers (min to max, thinner line)
        ax_stages.hlines(y=i, xmin=min_val, xmax=max_val, 
                         color=color, linewidth=1, alpha=0.5)
        
        # IQR box
        rect = mpatches.Rectangle((q25, i - box_height/2), q75 - q25, box_height, 
                                   facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.8)
        ax_stages.add_patch(rect)
        
        # Median line
        ax_stages.vlines(x=median, ymin=i - box_height/2, ymax=i + box_height/2, 
                         color='black', linewidth=2)
    
    ax_stages.set_yticks(y_positions)
    ax_stages.set_yticklabels(STAGE_ORDER, fontsize=9)
    ax_stages.set_xlabel('Diffusion Pseudotime', fontsize=10)
    ax_stages.set_xlim(0, 1)
    ax_stages.set_ylim(-0.5, len(STAGE_ORDER) - 0.5)
    ax_stages.invert_yaxis()  # Put early stages at top
    ax_stages.grid(True, axis='x', alpha=0.15)
    ax_stages.spines['top'].set_visible(False)
    ax_stages.spines['right'].set_visible(False)
    ax_stages.set_title('Disease Stage Distribution', fontsize=11, fontweight='bold', loc='left')
    
    fig.tight_layout()
    fig.savefig(appendix_dir / "cell_types_zscore_and_raw.png", bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {appendix_dir / 'cell_types_zscore_and_raw.png'}")


def generate_appendix_diagnostics(df: pd.DataFrame, base_plots_dir: Path):
    """
    Convenience wrapper.
    Writes into: <base_plots_dir>/appendix/
    """
    plot_pseudotime_density(df, base_plots_dir)
    plot_local_effective_sample_size(df, base_plots_dir, window_fraction=0.15)
    plot_smoothing_robustness(df, base_plots_dir)
    save_fibroblast_binned_bootstrap_figure(df, base_plots_dir, n_bins=25, n_boot=500, seed=0)
    save_stage_summary_csv(df, base_plots_dir)
    plot_cell_type_zscore_and_raw(df, base_plots_dir, window_fraction=0.15)
    print("generated appendix")


# =============================================================================
# In main(), after plot_panel_b_final(...), add:
# =============================================================================
# generate_appendix_diagnostics(df, plots_dir)


def main():
    parser = argparse.ArgumentParser()
    default_path = results_dir / "crc_conventional_uni2_2000_cellcounts.csv"
    parser.add_argument("--input", type=str, default=str(default_path))
    args = parser.parse_args()
    
    input_path = Path(args.input)
    print(f"📂 Loading: {input_path}")
    df = load_data(input_path)
    
    plot_panel_b_final(df, plots_dir / "standardized_abundance.png")

    generate_appendix_diagnostics(df, plots_dir)

if __name__ == "__main__":
    main()
# #!/usr/bin/env python
# """
# Paper Figure: Biological Validation (Panel B - Final v2).

# Changes:
# - Removed text labels from inside the plot.
# - Added a full-text "Stage Legend" below the X-axis.
# - Maintained the "box" visualization for distributions.

# Usage:
#     python plotting/figure_biological_validation_final_v2.py
# """

# import argparse
# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # =============================================================================
# # CONFIG
# # =============================================================================
# script_path = Path(__file__).resolve()
# project_root = script_path.parent.parent 
# results_dir = project_root / "results"
# plots_dir = project_root / "plots" / "biological_validation"
# plots_dir.mkdir(parents=True, exist_ok=True)

# COLORS = {
#     'Cancer Cells': '#D62828',       # Red
#     'Fibroblasts': '#8D6E63',        # Brown
#     'Apoptotic Bodies': '#9C27B0',   # Purple
#     'Neutrophils': '#F4A261',        # Orange
# }

# STAGE_COLORS = {
#     'Adenoma low grade': '#2ECC71',      # Green
#     'Adenoma high grade': '#F1C40F',     # Yellow
#     'Adenocarcinoma low grade': '#E67E22', # Orange
#     'Adenocarcinoma high grade': '#C0392B' # Red
# }

# STAGE_ORDER = [
#     'Adenoma low grade', 
#     'Adenoma high grade', 
#     'Adenocarcinoma low grade', 
#     'Adenocarcinoma high grade'
# ]

# def load_data(filepath):
#     if not filepath.exists():
#         print(f"❌ Error: File not found at {filepath}")
#         sys.exit(1)
#     df = pd.read_csv(filepath)
#     return df.sort_values('pseudotime_norm').reset_index(drop=True)

# def get_smoothed_data(df, columns, window_fraction=0.15):
#     window = int(len(df) * window_fraction)
#     smoothed = df[columns].rolling(window=window, center=True, min_periods=window//4).mean()
#     smoothed['pseudotime'] = df['pseudotime_norm']
#     return smoothed.dropna()

# def plot_panel_b_final(df, output_path):
#     print("🎨 Generating Panel B (Final v2)...")
    
#     # Increase height slightly to accommodate the bottom legend
#     fig, ax = plt.subplots(figsize=(5, 4.8), dpi=300)
    
#     # 1. Plot Z-Score Lines
#     z_features = {
#         'cancer_cells_prop': ('Cancer Cells', COLORS['Cancer Cells'], '--'),
#         'fibroblasts_prop': ('Fibroblasts', COLORS['Fibroblasts'], '-'),
#         'apoptotic_bodies_prop': ('Apoptosis', COLORS['Apoptotic Bodies'], '-'),
#         'neutrophils_prop': ('Neutrophils', COLORS['Neutrophils'], '-')
#     }
    
#     cols = list(z_features.keys())
#     smooth_z = get_smoothed_data(df, cols)
    
#     # Calculate Z-Scores
#     z_df = (smooth_z[cols] - smooth_z[cols].mean()) / smooth_z[cols].std()
#     z_df['pseudotime'] = smooth_z['pseudotime']
    
#     for col, (label, color, style) in z_features.items():
#         alpha = 0.5 if 'cancer' in col else 1.0
#         width = 2.0
#         ax.plot(z_df['pseudotime'], z_df[col], label=label, 
#                 color=color, linestyle=style, linewidth=width, alpha=alpha)

#     # 2. Add Stage Boxes (No Text Labels)
#     y_min, y_max = -2.5, 4.0
#     box_height = 0.25
#     box_y_pos = y_min + 0.1
    
#     for stage in STAGE_ORDER:
#         mask = df['class_label'] == stage
#         if not mask.any(): continue
        
#         pts = df.loc[mask, 'pseudotime_norm']
#         q25, q75 = pts.quantile(0.25), pts.quantile(0.75)
#         median = pts.median()
#         color = STAGE_COLORS.get(stage, 'gray')
        
#         # Box
#         rect = mpatches.Rectangle((q25, box_y_pos), q75 - q25, box_height, 
#                                   facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
#         ax.add_patch(rect)
#         # Median Line
#         ax.plot([median, median], [box_y_pos, box_y_pos + box_height], 
#                 color='black', linewidth=1.5)

#     # 3. Formatting
#     ax.set_ylabel('Standardized Abundance (Z-Score)', fontsize=10)
#     ax.set_xlabel('Diffusion Pseudotime', fontsize=10)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(y_min, y_max)
#     ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    
#     # Feature Legend (Top Left)
#     legend1 = ax.legend(loc='upper left', ncol=2, frameon=True, fontsize=8, 
#                         columnspacing=1.0, handlelength=1.5)
#     ax.add_artist(legend1) # Keep this legend when adding the next one

#     # 4. Create "Stage Key" Below X-Axis
#     stage_patches = [mpatches.Patch(facecolor=STAGE_COLORS[s], edgecolor='black', 
#                                     alpha=0.6, label=s) for s in STAGE_ORDER]
    
#     # Place this legend below the axis title
#     fig.legend(handles=stage_patches, loc='lower center', 
#                bbox_to_anchor=(0.5, 0.02), ncol=2, 
#                fontsize=8, frameon=False, title="")

#     ax.grid(True, alpha=0.15)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Adjust layout to make room for the bottom legend
#     plt.subplots_adjust(bottom=0.22)
    
#     plt.savefig(output_path, bbox_inches='tight')
#     plt.close()
#     print(f"✅ Saved: {output_path}")

# # =============================================================================
# # APPENDIX DIAGNOSTICS (drop-in)
# # =============================================================================

# def _ensure_appendix_dir(base_dir: Path) -> Path:
#     appendix_dir = base_dir / "appendix"
#     appendix_dir.mkdir(parents=True, exist_ok=True)
#     return appendix_dir


# def plot_pseudotime_density(df: pd.DataFrame, out_dir: Path, bins: int = 60):
#     """
#     A) Global pseudotime density overall + per class.
#     Saves:
#       - pseudotime_density_overall.png
#       - pseudotime_density_by_stage.png
#     """
#     appendix_dir = _ensure_appendix_dir(out_dir)

#     # Overall histogram
#     fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)
#     ax.hist(df["pseudotime_norm"].values, bins=bins)
#     ax.set_xlabel("Diffusion Pseudotime")
#     ax.set_ylabel("Count")
#     ax.set_title("Pseudotime density (all patches)")
#     ax.set_xlim(0, 1)
#     ax.grid(True, alpha=0.15)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     fig.tight_layout()
#     fig.savefig(appendix_dir / "pseudotime_density_overall.png", bbox_inches="tight")
#     plt.close(fig)

#     # Per-stage hist overlays (4 series)
#     fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)
#     for stage in STAGE_ORDER:
#         mask = df["class_label"] == stage
#         if not mask.any():
#             continue
#         ax.hist(
#             df.loc[mask, "pseudotime_norm"].values,
#             bins=bins,
#             alpha=0.5,
#             label=stage,
#         )
#     ax.set_xlabel("Diffusion Pseudotime")
#     ax.set_ylabel("Count")
#     ax.set_title("Pseudotime density by stage (hist overlays)")
#     ax.set_xlim(0, 1)
#     ax.legend(frameon=False, fontsize=7)
#     ax.grid(True, alpha=0.15)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     fig.tight_layout()
#     fig.savefig(appendix_dir / "pseudotime_density_by_stage.png", bbox_inches="tight")
#     plt.close(fig)


# def plot_local_effective_sample_size(
#     df: pd.DataFrame,
#     out_dir: Path,
#     window_fraction: float = 0.15,
#     bins: int = 80,
# ):
#     """
#     B) Local support: counts per pseudotime bin + "effective N" in a rolling window.

#     This aligns with your smoother, which uses a rolling window of:
#         window = int(len(df) * window_fraction)

#     Saves:
#       - pseudotime_local_counts_and_windowN.png
#     """
#     appendix_dir = _ensure_appendix_dir(out_dir)

#     pts = df["pseudotime_norm"].values
#     N = len(df)
#     window = max(5, int(N * window_fraction))

#     # Bin counts
#     edges = np.linspace(0.0, 1.0, bins + 1)
#     bin_centers = 0.5 * (edges[:-1] + edges[1:])
#     bin_counts, _ = np.histogram(pts, bins=edges)

#     # Rolling-window "effective N" at each bin center:
#     # count points within [center - w/2, center + w/2] in pseudotime.
#     # We approximate with a fixed pseudotime half-width set by the empirical window size.
#     pts_sorted = np.sort(pts)
#     # Approximate pseudotime half-width using median spacing * window/2
#     # (robust to uneven density, but still only an approximation).
#     if len(pts_sorted) >= 2:
#         median_dx = np.median(np.diff(pts_sorted))
#     else:
#         median_dx = 1.0
#     half_width = (window / 2.0) * median_dx

#     window_counts = []
#     for c in bin_centers:
#         lo = c - half_width
#         hi = c + half_width
#         # count in [lo, hi]
#         left = np.searchsorted(pts_sorted, lo, side="left")
#         right = np.searchsorted(pts_sorted, hi, side="right")
#         window_counts.append(right - left)
#     window_counts = np.asarray(window_counts)

#     fig, ax = plt.subplots(figsize=(5, 3.4), dpi=300)
#     ax.plot(bin_centers, bin_counts, label=f"Bin counts (bins={bins})")
#     ax.plot(bin_centers, window_counts, label=f"Approx. window support (N~{window})")
#     ax.set_xlabel("Diffusion Pseudotime")
#     ax.set_ylabel("Count")
#     ax.set_title("Local sampling support over pseudotime")
#     ax.set_xlim(0, 1)
#     ax.legend(frameon=False, fontsize=7)
#     ax.grid(True, alpha=0.15)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     fig.tight_layout()
#     fig.savefig(
#         appendix_dir / "pseudotime_local_counts_and_windowN.png",
#         bbox_inches="tight",
#     )
#     plt.close(fig)


# def plot_smoothing_robustness(
#     df: pd.DataFrame,
#     out_dir: Path,
#     columns=None,
#     window_fractions=(0.10, 0.15, 0.20),
# ):
#     """
#     C) Robustness to smoothing: overlay smoothed Z-score curves under multiple window fractions.

#     Saves:
#       - smoothing_robustness_fibroblasts.png
#       - smoothing_robustness_all_features.png
#     """
#     appendix_dir = _ensure_appendix_dir(out_dir)

#     if columns is None:
#         columns = [
#             "cancer_cells_prop",
#             "fibroblasts_prop",
#             "apoptotic_bodies_prop",
#             "neutrophils_prop",
#         ]

#     # --- (1) Fibroblast-only robustness (most relevant to "V-shape" claims) ---
#     fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)
#     for wf in window_fractions:
#         smooth = get_smoothed_data(df, ["fibroblasts_prop"], window_fraction=wf)
#         z = (smooth["fibroblasts_prop"] - smooth["fibroblasts_prop"].mean()) / smooth[
#             "fibroblasts_prop"
#         ].std()
#         ax.plot(smooth["pseudotime"], z.values, label=f"window_fraction={wf:.2f}")
#     ax.set_xlabel("Diffusion Pseudotime")
#     ax.set_ylabel("Fibroblasts (Z-score)")
#     ax.set_title("Smoothing robustness (fibroblasts)")
#     ax.set_xlim(0, 1)
#     ax.legend(frameon=False, fontsize=7)
#     ax.grid(True, alpha=0.15)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     fig.tight_layout()
#     fig.savefig(appendix_dir / "smoothing_robustness_fibroblasts.png", bbox_inches="tight")
#     plt.close(fig)

#     # --- (2) All-feature robustness (small multiples in one panel is tempting,
#     # but keeping it readable: overlay each feature separately in the same axis via linestyle)
#     # If you prefer one-feature-per-plot, tell me and I'll adjust.
#     feature_specs = {
#         "cancer_cells_prop": ("Cancer Cells", COLORS.get("Cancer Cells", None), "--"),
#         "fibroblasts_prop": ("Fibroblasts", COLORS.get("Fibroblasts", None), "-"),
#         "apoptotic_bodies_prop": ("Apoptosis", COLORS.get("Apoptotic Bodies", None), "-"),
#         "neutrophils_prop": ("Neutrophils", COLORS.get("Neutrophils", None), "-"),
#     }

#     fig, ax = plt.subplots(figsize=(5, 3.4), dpi=300)
#     for wf in window_fractions:
#         smooth = get_smoothed_data(df, columns, window_fraction=wf)
#         z = (smooth[columns] - smooth[columns].mean()) / smooth[columns].std()
#         for col, (label, color, style) in feature_specs.items():
#             if col not in columns:
#                 continue
#             alpha = 0.35 if "cancer" in col else 0.6
#             ax.plot(
#                 smooth["pseudotime"],
#                 z[col].values,
#                 linestyle=style,
#                 alpha=alpha,
#                 label=f"{label} (wf={wf:.2f})",
#             )
#     ax.set_xlabel("Diffusion Pseudotime")
#     ax.set_ylabel("Z-score")
#     ax.set_title("Smoothing robustness (all features)")
#     ax.set_xlim(0, 1)
#     ax.grid(True, alpha=0.15)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     # Make legend compact
#     ax.legend(
#         frameon=False,
#         fontsize=6,
#         ncol=2,
#         columnspacing=0.8,
#         handlelength=1.4,
#         loc="upper left",
#     )

#     fig.tight_layout()
#     fig.savefig(appendix_dir / "smoothing_robustness_all_features.png", bbox_inches="tight")
#     plt.close(fig)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def save_fibroblast_binned_bootstrap_figure(
#     df: pd.DataFrame,
#     out_dir: Path,
#     n_bins: int = 25,
#     n_boot: int = 500,
#     seed: int = 0,
#     min_n_visible: int = 50,
# ):
#     """
#     Create a binned fibroblast vs pseudotime figure with bootstrap CIs.
#     Saves a PNG into <out_dir>/appendix/.

#     Figure shows:
#       - mean fibroblast proportion per bin
#       - 90% bootstrap CI (shaded)
#       - bins with low support (n < min_n_visible) faded
#     """
#     appendix_dir = _ensure_appendix_dir(out_dir)

#     d = df[["pseudotime_norm", "fibroblasts_prop"]].dropna().copy()
#     d = d.sort_values("pseudotime_norm").reset_index(drop=True)

#     # Bin pseudotime
#     edges = np.linspace(0.0, 1.0, n_bins + 1)
#     centers = 0.5 * (edges[:-1] + edges[1:])
#     bin_id = np.digitize(d["pseudotime_norm"].values, edges, right=False) - 1
#     bin_id = np.clip(bin_id, 0, n_bins - 1)

#     rng = np.random.default_rng(seed)

#     means, ci5s, ci95s, ns = [], [], [], []

#     for b in range(n_bins):
#         mask = bin_id == b
#         y = d.loc[mask, "fibroblasts_prop"].values
#         n = int(mask.sum())
#         ns.append(n)

#         if n == 0:
#             means.append(np.nan)
#             ci5s.append(np.nan)
#             ci95s.append(np.nan)
#             continue

#         mean = np.mean(y)
#         means.append(mean)

#         if n_boot > 0 and n > 1:
#             boot_means = np.empty(n_boot)
#             for i in range(n_boot):
#                 samp = rng.choice(y, size=n, replace=True)
#                 boot_means[i] = np.mean(samp)
#             ci5, ci95 = np.quantile(boot_means, [0.05, 0.95])
#         else:
#             ci5, ci95 = np.nan, np.nan

#         ci5s.append(ci5)
#         ci95s.append(ci95)

#     means = np.asarray(means)
#     ci5s = np.asarray(ci5s)
#     ci95s = np.asarray(ci95s)
#     ns = np.asarray(ns)

#     # Plot
#     fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)

#     # CI band
#     ax.fill_between(
#         centers,
#         ci5s,
#         ci95s,
#         color="#8D6E63",  # fibroblast brown
#         alpha=0.25,
#         linewidth=0,
#         label="90% bootstrap CI",
#     )

#     # Mean line
#     ax.plot(
#         centers,
#         means,
#         color="#8D6E63",
#         linewidth=2.0,
#         label="Binned mean",
#     )

#     # Fade low-support bins
#     low_support = ns < min_n_visible
#     ax.scatter(
#         centers[low_support],
#         means[low_support],
#         color="#8D6E63",
#         alpha=0.35,
#         s=20,
#         zorder=3,
#     )

#     ax.set_xlabel("Diffusion Pseudotime")
#     ax.set_ylabel("Fibroblast Proportion")
#     ax.set_xlim(0, 1)
#     ax.grid(True, alpha=0.15)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.legend(frameon=False, fontsize=8)

#     fig.tight_layout()
#     fig.savefig(
#         appendix_dir / "fibroblasts_binned_bootstrap.png",
#         bbox_inches="tight",
#     )
#     plt.close(fig)



# def save_stage_summary_csv(df: pd.DataFrame, out_dir: Path):
#     """
#     (2) Stage-level summary CSV for appendix:
#         - n_patches
#         - pseudotime median/IQR
#         - fibroblasts_prop median/IQR
#         - fibroblasts_prop mean/std

#     Output in <out_dir>/appendix/:
#       - stage_summary_fibroblasts.csv
#     """
#     appendix_dir = _ensure_appendix_dir(out_dir)

#     d = df[["class_label", "pseudotime_norm", "fibroblasts_prop"]].dropna().copy()

#     def iqr(x):
#         q25, q75 = np.quantile(x, [0.25, 0.75])
#         return float(q25), float(q75)

#     rows = []
#     for stage in STAGE_ORDER:
#         s = d[d["class_label"] == stage]
#         if len(s) == 0:
#             continue

#         pt = s["pseudotime_norm"].values
#         fb = s["fibroblasts_prop"].values

#         pt_q25, pt_q75 = iqr(pt)
#         fb_q25, fb_q75 = iqr(fb)

#         rows.append(
#             {
#                 "stage": stage,
#                 "n_patches": int(len(s)),
#                 "pseudotime_median": float(np.median(pt)),
#                 "pseudotime_q25": pt_q25,
#                 "pseudotime_q75": pt_q75,
#                 "fibroblasts_prop_median": float(np.median(fb)),
#                 "fibroblasts_prop_q25": fb_q25,
#                 "fibroblasts_prop_q75": fb_q75,
#                 "fibroblasts_prop_mean": float(np.mean(fb)),
#                 "fibroblasts_prop_std": float(np.std(fb, ddof=1)) if len(fb) > 1 else np.nan,
#             }
#         )

#     out = pd.DataFrame(rows)
#     out_path = appendix_dir / "stage_summary_fibroblasts.csv"
#     out.to_csv(out_path, index=False)

# def generate_appendix_diagnostics(df: pd.DataFrame, base_plots_dir: Path):
#     """
#     Convenience wrapper.
#     Writes into: <base_plots_dir>/appendix/
#     """
#     plot_pseudotime_density(df, base_plots_dir)
#     plot_local_effective_sample_size(df, base_plots_dir, window_fraction=0.15)
#     plot_smoothing_robustness(df, base_plots_dir)
#     save_fibroblast_binned_bootstrap_figure(df, base_plots_dir, n_bins=25, n_boot=500, seed=0)
#     save_stage_summary_csv(df, base_plots_dir)
#     print("generated appendix")


# # =============================================================================
# # In main(), after plot_panel_b_final(...), add:
# # =============================================================================
# # generate_appendix_diagnostics(df, plots_dir)


# def main():
#     parser = argparse.ArgumentParser()
#     default_path = results_dir / "crc_conventional_uni2_2000_cellcounts.csv"
#     parser.add_argument("--input", type=str, default=str(default_path))
#     args = parser.parse_args()
    
#     input_path = Path(args.input)
#     print(f"📂 Loading: {input_path}")
#     df = load_data(input_path)
    
#     plot_panel_b_final(df, plots_dir / "standardized_abundance.png")

#     generate_appendix_diagnostics(df, plots_dir)

# if __name__ == "__main__":
#     main()