#!/usr/bin/env python
"""
Paper Figures: Biological Validation of Pseudotime via Cell Composition.

Main figure: Clean summary showing key cell type trends across progression
Supplementary: Full stacked panel + summary table

Narrative framing:
- Primary: Cell composition validates pseudotime aligns with known biology
- Secondary: Within-stage gradients suggest pseudotime captures sub-stage severity

Usage:
    python plotting/figure_biological_validation.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

# Path setup
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import config
    RESULTS_DIR = config.RESULTS_RUN_DIR
    PLOTS_DIR = config.PLOTS_OUTPUT_DIR if hasattr(config, 'PLOTS_OUTPUT_DIR') else Path("plots")
except ImportError:
    RESULTS_DIR = project_root / "results"
    PLOTS_DIR = project_root / "plots"

# =============================================================================
# CONFIGURATION
# =============================================================================

STAGE_ORDER = [
    'Adenoma low grade', 
    'Adenoma high grade', 
    'Adenocarcinoma low grade', 
    'Adenocarcinoma high grade'
]

STAGE_COLORS = {
    'Adenoma low grade': '#2ECC71',
    'Adenoma high grade': '#F1C40F', 
    'Adenocarcinoma low grade': '#E67E22',
    'Adenocarcinoma high grade': '#E74C3C',
}

STAGE_SHORT = {
    'Adenoma low grade': 'Adenoma\n(low)',
    'Adenoma high grade': 'Adenoma\n(high)', 
    'Adenocarcinoma low grade': 'Carcinoma\n(low)',
    'Adenocarcinoma high grade': 'Carcinoma\n(high)',
}

# Key cell types for main figure (the interpretable ones)
KEY_CELL_TYPES = [
    ('cancer_cells', 'Cancer Cells', '#E63946'),
    ('epithelial_normal', 'Normal Epithelium', '#06D6A0'),
    ('macrophages', 'Macrophages', '#9D4EDD'),
    ('neutrophils', 'Neutrophils', '#F4A261'),
    ('lymphocytes', 'Lymphocytes', '#457B9D'),
]


def smooth_line(x, y, window=80):
    """Compute smoothed trend line."""
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    y_smooth = uniform_filter1d(y_sorted.astype(float), size=window, mode='nearest')
    return x_sorted, y_smooth


def get_stage_boundaries(df: pd.DataFrame):
    """Get pseudotime ranges for each stage."""
    boundaries = {}
    for stage in STAGE_ORDER:
        stage_df = df[df['class_label'] == stage]
        pt = stage_df['pseudotime_norm']
        boundaries[stage] = {
            'q25': pt.quantile(0.25),
            'q75': pt.quantile(0.75),
            'min': pt.min(),
            'max': pt.max(),
            'median': pt.median(),
        }
    return boundaries


def compute_stage_means(df: pd.DataFrame, cell_types: list) -> pd.DataFrame:
    """Compute mean cell proportions by stage."""
    results = []
    for stage in STAGE_ORDER:
        stage_df = df[df['class_label'] == stage]
        row = {'stage': stage}
        for ct, _, _ in cell_types:
            col = f"{ct}_prop"
            if col in df.columns:
                row[ct] = stage_df[col].mean() * 100
        results.append(row)
    return pd.DataFrame(results)


# =============================================================================
# BAR CHART FIGURE: Cell composition by stage (single-column version)
# =============================================================================

# Extended cell types for bar chart
BAR_CHART_CELL_TYPES = [
    ('cancer_cells', 'Cancer Cells', '#E63946'),
    ('epithelial_normal', 'Normal Epithelium', '#06D6A0'),
    ('fibroblasts', 'Fibroblasts', '#8D6E63'),
    ('lymphocytes', 'Lymphocytes', '#457B9D'),
    ('macrophages', 'Macrophages', '#9D4EDD'),
    ('neutrophils', 'Neutrophils', '#F4A261'),
    ('plasmocytes', 'Plasmocytes', '#E91E63'),
    ('endothelial', 'Endothelial', '#00BCD4'),
    ('apoptotic_bodies', 'Apoptotic Bodies', '#607D8B'),
]

# Short stage names for compact display
STAGE_SHORT_COMPACT = {
    'Adenoma low grade': 'Ad-LG',
    'Adenoma high grade': 'Ad-HG', 
    'Adenocarcinoma low grade': 'Ca-LG',
    'Adenocarcinoma high grade': 'Ca-HG',
}


def create_bar_chart_figure(df: pd.DataFrame, output_path: Path):
    """
    Standalone bar chart showing mean cell composition by stage.
    Single-column friendly version for ICML.
    """
    
    # Smaller figure for single column (ICML column width ~3.25 inches)
    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    
    stage_means = compute_stage_means(df, BAR_CHART_CELL_TYPES)
    
    x = np.arange(len(STAGE_ORDER))
    n_types = len(BAR_CHART_CELL_TYPES)
    width = 0.8 / n_types
    
    for i, (ct, label, color) in enumerate(BAR_CHART_CELL_TYPES):
        if ct in stage_means.columns:
            values = stage_means[ct].values
            offset = (i - n_types/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=label, color=color, 
                         edgecolor='white', linewidth=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_SHORT_COMPACT[s] for s in STAGE_ORDER], fontsize=8)
    ax.set_ylabel('Mean % of Cells', fontsize=9)
    # ax.set_xlabel('Disease Stage', fontsize=9)
    
    # Compact legend outside plot
    ax.legend(fontsize=6, loc='upper right', framealpha=0.95, 
              ncol=1, handlelength=1, handletextpad=0.5,
              columnspacing=0.5, labelspacing=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, None)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=8)
    
    # Smaller progression arrow
    # ax.annotate('', xy=(3.4, -0.06), xytext=(-0.4, -0.06),
    #             xycoords=('data', 'axes fraction'), 
    #             textcoords=('data', 'axes fraction'),
    #             arrowprops=dict(arrowstyle='->', color='gray', lw=1),
    #             annotation_clip=False)
    # ax.text(1.5, -0.10, 'Progression', ha='center', fontsize=7, 
    #         color='gray', style='italic',
    #         transform=ax.get_xaxis_transform())
    
    # ax.set_title('Cell Composition by Stage', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Bar chart figure saved: {output_path}")


def create_bar_chart_figure_wide(df: pd.DataFrame, output_path: Path):
    """
    Wide bar chart for two-column display.
    """
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    stage_means = compute_stage_means(df, BAR_CHART_CELL_TYPES)
    
    x = np.arange(len(STAGE_ORDER))
    n_types = len(BAR_CHART_CELL_TYPES)
    width = 0.8 / n_types
    
    for i, (ct, label, color) in enumerate(BAR_CHART_CELL_TYPES):
        if ct in stage_means.columns:
            values = stage_means[ct].values
            offset = (i - n_types/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=label, color=color, 
                         edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_SHORT[s] for s in STAGE_ORDER], fontsize=9)
    ax.set_ylabel('Mean % of Cells', fontsize=10)
    # ax.set_xlabel('Disease Stage', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.95, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, None)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    ax.annotate('', xy=(3.5, -0.08), xytext=(-0.5, -0.08),
                xycoords=('data', 'axes fraction'), 
                textcoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2),
                annotation_clip=False)
    ax.text(1.5, -0.12, 'Disease Progression', ha='center', fontsize=9, 
            color='gray', style='italic',
            transform=ax.get_xaxis_transform())
    
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Wide bar chart figure saved: {output_path}")


# =============================================================================
# SIMPLE FIGURE: Trend lines with stage box plots at bottom
# =============================================================================

def create_main_figure_simple(df: pd.DataFrame, output_path: Path):
    """
    Clean two-panel figure:
    - Top: Trend lines for key cell types vs pseudotime (white background)
    - Bottom: Box plots showing stage distribution along pseudotime
    """
    
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)
    
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    
    stage_bounds = get_stage_boundaries(df)
    df_sorted = df.sort_values('pseudotime_norm')
    x_pt = df_sorted['pseudotime_norm'].values
    
    # ==========================================================================
    # TOP PANEL: Trend lines (white background - no stage coloring)
    # ==========================================================================
    
    for ct, label, color in KEY_CELL_TYPES:
        col = f"{ct}_prop"
        if col not in df.columns:
            continue
        
        y = df_sorted[col].values
        x_smooth, y_smooth = smooth_line(x_pt, y, window=80)
        
        ax_top.plot(x_smooth, y_smooth * 100, color=color, linewidth=2.5, label=label)
    
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 100)
    ax_top.set_ylabel('Proportion of Cells (%)', fontsize=11)
    ax_top.legend(fontsize=9, loc='center right', framealpha=0.95)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax_top.get_xticklabels(), visible=False)
    
    # ax_top.set_title('Cellular Composition Along Disease Progression',
    #                  fontsize=12, fontweight='bold')
    
    # ==========================================================================
    # BOTTOM PANEL: Box plots for stage distribution
    # ==========================================================================
    
    box_data = []
    box_colors = []
    
    for stage in STAGE_ORDER:
        stage_data = df[df['class_label'] == stage]['pseudotime_norm'].values
        box_data.append(stage_data)
        box_colors.append(STAGE_COLORS[stage])
    
    # Create horizontal box plots
    bp = ax_bottom.boxplot(
        box_data,
        positions=range(len(STAGE_ORDER)),
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        whiskerprops=dict(color='gray', linewidth=1),
        capprops=dict(color='gray', linewidth=1),
        medianprops=dict(color='black', linewidth=2),
    )
    
    # Color only the box portion
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Stage labels on y-axis
    ax_bottom.set_yticks(range(len(STAGE_ORDER)))
    ax_bottom.set_yticklabels(STAGE_ORDER, fontsize=9)
    ax_bottom.set_xlabel('Pseudotime (Normalized)', fontsize=11)
    ax_bottom.set_xlim(0, 1)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)
    ax_bottom.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add vertical lines at box edges to connect panels visually
    for i, stage in enumerate(STAGE_ORDER):
        bounds = stage_bounds[stage]
        # Light vertical spans in top panel at IQR
        ax_top.axvspan(bounds['q25'], bounds['q75'], 
                      color=STAGE_COLORS[stage], alpha=0.08, zorder=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Simple main figure saved: {output_path}")


# =============================================================================
# SUPPLEMENTARY TABLE: Summary statistics
# =============================================================================

def create_summary_table(df: pd.DataFrame, output_path: Path):
    """
    Create summary table for supplementary materials.
    Shows mean ± std by stage (all 4 stages), overall correlation with pseudotime,
    and within-stage correlations.
    """
    
    all_cell_types = [
        "cancer_cells", "epithelial_normal", "fibroblasts", "lymphocytes",
        "macrophages", "plasmocytes", "neutrophils", "eosinophils",
        "endothelial", "apoptotic_bodies", "smooth_muscle", "red_blood_cells",
        "mitotic_figures"
    ]
    
    rows = []
    
    for ct in all_cell_types:
        col = f"{ct}_prop"
        if col not in df.columns:
            continue
        
        row = {'Cell Type': ct.replace('_', ' ').title()}
        
        # Overall correlation with pseudotime
        rho, pval = spearmanr(df['pseudotime_norm'], df[col])
        row['Overall ρ'] = f"{rho:+.3f}"
        row['Overall p'] = f"{pval:.2e}"
        
        # Mean by stage (ALL 4 STAGES)
        for stage in STAGE_ORDER:
            stage_df = df[df['class_label'] == stage]
            mean = stage_df[col].mean() * 100
            std = stage_df[col].std() * 100
            # Use short name for column
            short_name = stage.replace('Adenoma ', 'Ad-').replace('Adenocarcinoma ', 'Ca-').replace(' grade', '')
            row[short_name] = f"{mean:.1f}±{std:.1f}"
        
        # Within-stage correlations
        within_rhos = []
        for stage in STAGE_ORDER:
            stage_df = df[df['class_label'] == stage]
            rho_w, pval_w = spearmanr(stage_df['pseudotime_norm'], stage_df[col])
            sig = '*' if pval_w < 0.05 else ''
            within_rhos.append(f"{rho_w:+.2f}{sig}")
        
        row['Within-stage ρ'] = ' | '.join(within_rhos)
        
        rows.append(row)
    
    table_df = pd.DataFrame(rows)
    table_df.to_csv(output_path, index=False)
    
    print(f"✅ Summary table saved: {output_path}")
    
    # Also print for inspection
    print("\n" + "=" * 100)
    print("SUPPLEMENTARY TABLE: Cell Composition Summary (All 4 Stages)")
    print("=" * 100)
    print(table_df.to_string(index=False))
    print("=" * 100 + "\n")
    
    # Print stage differences for middle stages
    print("\n📊 MIDDLE STAGE COMPARISON (Adenoma HG vs Adenocarcinoma LG):")
    print("-" * 60)
    
    for ct in all_cell_types:
        col = f"{ct}_prop"
        if col not in df.columns:
            continue
        
        ad_hg = df[df['class_label'] == 'Adenoma high grade'][col].mean() * 100
        ca_lg = df[df['class_label'] == 'Adenocarcinoma low grade'][col].mean() * 100
        diff = ca_lg - ad_hg
        
        if abs(diff) > 0.5:  # Only show meaningful differences
            arrow = '↑' if diff > 0 else '↓'
            print(f"  {ct:<20}: Ad-HG={ad_hg:5.1f}% → Ca-LG={ca_lg:5.1f}%  ({arrow}{abs(diff):+.1f}%)")
    
    print("-" * 60)
    
    return table_df


# =============================================================================
# SUPPLEMENTARY FIGURE: Full stacked panel (same as before)
# =============================================================================

def create_supplementary_stacked(df: pd.DataFrame, output_path: Path):
    """
    Full stacked panel figure for supplementary materials.
    All cell types with stage backgrounds and within-stage correlations.
    """
    
    all_cell_types = [
        ("cancer_cells", "Cancer Cells"),
        ("epithelial_normal", "Normal Epithelium"),
        ("fibroblasts", "Fibroblasts"),
        ("lymphocytes", "Lymphocytes"),
        ("macrophages", "Macrophages"),
        ("plasmocytes", "Plasmocytes"),
        ("neutrophils", "Neutrophils"),
        ("eosinophils", "Eosinophils"),
        ("endothelial", "Endothelial"),
        ("apoptotic_bodies", "Apoptotic Bodies"),
        ("smooth_muscle", "Smooth Muscle"),
        ("red_blood_cells", "Red Blood Cells"),
        ("mitotic_figures", "Mitotic Figures"),
    ]
    
    # Filter to existing
    cell_types = [(ct, name) for ct, name in all_cell_types if f"{ct}_prop" in df.columns]
    n_panels = len(cell_types)
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 1.2 * n_panels), sharex=True)
    
    stage_bounds = get_stage_boundaries(df)
    df_sorted = df.sort_values('pseudotime_norm')
    x = df_sorted['pseudotime_norm'].values
    
    for idx, (ax, (ct, name)) in enumerate(zip(axes, cell_types)):
        col = f"{ct}_prop"
        y = df_sorted[col].values
        
        # Stage backgrounds
        for stage in STAGE_ORDER:
            bounds = stage_bounds[stage]
            ax.axvspan(bounds['min'], bounds['max'], 
                      color=STAGE_COLORS[stage], alpha=0.2, zorder=0)
            ax.axvspan(bounds['q25'], bounds['q75'],
                      color=STAGE_COLORS[stage], alpha=0.25, zorder=1)
        
        # Trend line
        x_smooth, y_smooth = smooth_line(x, y, window=80)
        ax.plot(x_smooth, y_smooth, color='#2C3E50', linewidth=2, zorder=10)
        
        # Within-stage correlations
        rho_parts = []
        for stage in STAGE_ORDER:
            stage_df = df[df['class_label'] == stage]
            rho, pval = spearmanr(stage_df['pseudotime_norm'], stage_df[col])
            sig = '*' if pval < 0.05 else ''
            arrow = '↑' if rho > 0.05 else ('↓' if rho < -0.05 else '→')
            rho_parts.append(f"{arrow}{rho:+.2f}{sig}")
        
        ax.text(1.02, 0.5, '  '.join(rho_parts), transform=ax.transAxes,
               fontsize=7, va='center', ha='left', family='monospace', color='#555')
        
        # Formatting
        ax.set_ylabel(name, fontsize=9, rotation=0, ha='right', va='center', labelpad=60)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(0.15, y.max() * 1.1))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.3, linestyle=':', zorder=0)
        
        if idx < n_panels - 1:
            ax.tick_params(axis='x', labelbottom=False)
    
    axes[-1].set_xlabel('Pseudotime (Normalized)', fontsize=11)
    
    # Legend
    legend_handles = [Rectangle((0,0), 1, 1, facecolor=STAGE_COLORS[s], alpha=0.4) 
                     for s in STAGE_ORDER]
    legend_labels = [s.split()[0] + ' ' + s.split()[1][0].upper() + s.split()[1][1:3] 
                    for s in STAGE_ORDER]
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, 
              fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=False)
    
    # Header for rho columns
    header = '    '.join([s[:4] for s in STAGE_ORDER])
    axes[0].text(1.02, 1.2, header, transform=axes[0].transAxes,
                fontsize=7, va='bottom', ha='left', family='monospace',
                color='#888', fontweight='bold')
    
    fig.suptitle('Supplementary: All Cell Types Along Pseudotime\n'
                 '(ρ = within-stage Spearman correlation, *p<0.05)',
                 fontsize=11, fontweight='bold', y=1.06)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Supplementary stacked figure saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    input_path = Path(args.input) if args.input else RESULTS_DIR / "crc_conventional_uni2_2000_cellcounts.csv"
    output_dir = Path(args.output_dir) if args.output_dir else PLOTS_DIR / "cell_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   {len(df)} patches loaded")
    
    base_name = input_path.stem.replace('_cellcounts', '')
    
    # Main figures
    create_bar_chart_figure(df, output_dir / f"{base_name}_bar_chart.png")  # Single column
    create_bar_chart_figure_wide(df, output_dir / f"{base_name}_bar_chart_wide.png")  # Two column
    create_main_figure_simple(df, output_dir / f"{base_name}_trajectory.png")
    
    # Supplementary materials
    create_summary_table(df, output_dir / f"{base_name}_supp_table.csv")
    create_supplementary_stacked(df, output_dir / f"{base_name}_supp_stacked.png")
    
    print(f"\n✅ All figures saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - {base_name}_bar_chart.png (single-column bar chart)")
    print(f"  - {base_name}_bar_chart_wide.png (two-column bar chart)")
    print(f"  - {base_name}_trajectory.png (trends + stage box plots)")
    print(f"  - {base_name}_supp_table.csv (supplementary statistics)")
    print(f"  - {base_name}_supp_stacked.png (supplementary full figure)")


if __name__ == "__main__":
    main()