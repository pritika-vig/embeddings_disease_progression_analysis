#!/usr/bin/env python
"""
Plotting Script: Manifold Visualization for Permutation Test Section

Creates a figure showing:
- Panel A: Embedding space colored by pseudotime with stage centroids
- Panel B: Pseudotime distributions by stage (showing ordering is recovered)

This accompanies the permutation test boxplots to show WHY the biological
ordering achieves higher tau - the pseudotime genuinely follows progression.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.stats import kendalltau
import scanpy as sc

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import (
    DPTConfig,
    cohort_to_anndata,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
pcfg.set_icml_style()

TARGET_PROGRESSION = "SCC"  # Can also use "CRC-Conventional", "SCC", "CRC-Serrated"
TARGET_MODEL = "uni2"
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Stage colors - consistent palette
STAGE_COLORS = ["#3274A1", "#E1812C", "#9467BD", "#59A14F"]  # Blue, Orange, Purple, Green


def get_real_data():
    """Load embeddings and compute diffusion pseudotime."""
    prog_config = next(p for p in config.PROGRESSIONS if p["name"] == TARGET_PROGRESSION)
    biological_order = prog_config["classes"]
    
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=biological_order,
        models=[TARGET_MODEL],
        progression_name=TARGET_PROGRESSION,
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    patch_ids = dataset.sample_patch_ids(n_per_class=1000, max_per_slide=50, seed=42)
    
    dataset.load_model_into_memory(TARGET_MODEL, patch_ids, "final_embedding")
    cohort_df = dataset.get_cohort(patch_ids, TARGET_MODEL, "final_embedding")
    
    adata = cohort_to_anndata(cohort_df, biological_order)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=30, metric="cosine")
    sc.tl.diffmap(adata, n_comps=15)
    
    # Set root cell for DPT (earliest stage)
    early_mask = adata.obs['class'] == biological_order[0]
    early_indices = np.where(early_mask)[0]
    
    # Use the cell closest to the centroid of the earliest class
    X_diff = adata.obsm['X_diffmap']
    start_idx = 1 if np.std(X_diff[:, 0]) < 1e-5 else 0
    early_embeddings = X_diff[early_mask, start_idx:start_idx+3]
    centroid = np.median(early_embeddings, axis=0)
    distances = np.linalg.norm(early_embeddings - centroid, axis=1)
    root_idx = early_indices[np.argmin(distances)]
    
    adata.uns['iroot'] = root_idx
    sc.tl.dpt(adata)
    
    # Extract what we need
    dcs = X_diff[:, start_idx:start_idx+10]
    pseudotime = adata.obs['dpt_pseudotime'].values
    classes = adata.obs['class'].values
    
    # Normalize pseudotime to [0, 1]
    pseudotime = (pseudotime - np.nanmin(pseudotime)) / (np.nanmax(pseudotime) - np.nanmin(pseudotime))
    
    # Compute centroids
    centroids = {}
    for stage in biological_order:
        mask = classes == stage
        if mask.sum() > 0:
            centroids[stage] = np.median(dcs[mask], axis=0)
    
    return dcs, centroids, biological_order, pseudotime, classes


def rotate_to_progression_axis(X, centroids, stages):
    """
    Rotates space so Start -> End is the X-axis.
    Returns 2D projection for visualization.
    """
    start_node = centroids[stages[0]]
    end_node = centroids[stages[-1]]
    mid_node = centroids[stages[1]]
    
    # Axis 1 (u): Start -> End
    vec_main = end_node - start_node
    u = vec_main / np.linalg.norm(vec_main)
    
    # Axis 2 (v): Orthogonal part of Start -> Mid
    vec_mid = mid_node - start_node
    vec_mid_ortho = vec_mid - np.dot(vec_mid, u) * u
    if np.linalg.norm(vec_mid_ortho) > 1e-10:
        v = vec_mid_ortho / np.linalg.norm(vec_mid_ortho)
    else:
        # Fallback: use a random orthogonal vector
        v = np.zeros_like(u)
        v[0] = -u[1]
        v[1] = u[0]
        v = v / np.linalg.norm(v)
    
    # Project all points
    X_proj = np.column_stack([np.dot(X - start_node, u), np.dot(X - start_node, v)])
    
    # Project centroids
    centroids_proj = {}
    for stage in stages:
        vec = centroids[stage] - start_node
        centroids_proj[stage] = np.array([np.dot(vec, u), np.dot(vec, v)])
    
    return X_proj, centroids_proj


def draw_manifold_panel(ax, X_2d, centroids_2d, stages, pseudotime, classes):
    """
    Panel A: Embedding space colored by pseudotime with connected centroids.
    """
    n_stages = len(stages)
    colors = STAGE_COLORS[:n_stages]
    
    # Sort by pseudotime for better visual layering
    valid_mask = ~np.isnan(pseudotime)
    X_valid = X_2d[valid_mask]
    pt_valid = pseudotime[valid_mask]
    sort_idx = np.argsort(pt_valid)
    
    # Subsample for performance
    n_show = min(3000, len(sort_idx))
    show_idx = np.linspace(0, len(sort_idx)-1, n_show, dtype=int)
    
    # Scatter plot colored by pseudotime
    scatter = ax.scatter(X_valid[sort_idx[show_idx], 0], 
                         X_valid[sort_idx[show_idx], 1],
                         c=pt_valid[sort_idx[show_idx]], 
                         cmap='viridis', s=15, alpha=0.5,
                         edgecolors='none', rasterized=True)
    
    # Draw path through centroids
    centroid_pts = np.array([centroids_2d[s] for s in stages])
    ax.plot(centroid_pts[:, 0], centroid_pts[:, 1], 'k-', lw=3, zorder=5, alpha=0.9)
    
    # Draw centroid markers with stage numbers
    for i, stage in enumerate(stages):
        pt = centroids_2d[stage]
        
        # White background circle
        ax.scatter(pt[0], pt[1], c='white', s=450, edgecolors='black',
                   linewidth=2, zorder=10)
        # Colored inner circle
        ax.scatter(pt[0], pt[1], c=colors[i], s=350, edgecolors='white',
                   linewidth=1.5, zorder=11)
        # Stage number
        ax.text(pt[0], pt[1], str(i+1), ha='center', va='center',
                fontweight='bold', fontsize=12, color='white', zorder=12)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cbar.set_label('Diffusion Pseudotime', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Axis labels
    ax.set_xlabel('Progression Axis', fontsize=11)
    ax.set_ylabel('Deviation Axis', fontsize=11)
    ax.tick_params(labelsize=9)
    
    # Legend for stage markers
    legend_elements = [
        plt.scatter([], [], c=colors[i], s=120, edgecolors='white', linewidth=1,
                    label=f'Stage {i+1}')
        for i in range(n_stages)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9, title='Centroids', title_fontsize=9)
    
    ax.set_title(f"A. Embedding Space: {TARGET_MODEL.upper()} on {TARGET_PROGRESSION}",
                 fontweight='bold', loc='left', fontsize=12)


def draw_pseudotime_panel(ax, stages, pseudotime, classes):
    """
    Panel B: Violin plots showing pseudotime distribution by stage.
    """
    n_stages = len(stages)
    colors = STAGE_COLORS[:n_stages]
    
    # Convert class names to indices
    class_to_idx = {s: i for i, s in enumerate(stages)}
    class_indices = np.array([class_to_idx.get(c, -1) for c in classes])
    
    # Filter valid data
    valid_mask = (~np.isnan(pseudotime)) & (class_indices >= 0)
    pt_valid = pseudotime[valid_mask]
    idx_valid = class_indices[valid_mask]
    
    # Prepare data for violin plot
    data_by_stage = [pt_valid[idx_valid == i] for i in range(n_stages)]
    
    # Create violin plot
    parts = ax.violinplot(data_by_stage, positions=range(n_stages),
                          showmeans=False, showmedians=True, widths=0.7)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    # Style median lines
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cbars'].set_linewidth(1)
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')
    
    # Formatting
    ax.set_xticks(range(n_stages))
    ax.set_xticklabels([f"Stage {i+1}" for i in range(n_stages)], fontsize=10)
    ax.set_ylabel('Diffusion Pseudotime', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    
    # Trend line through medians
    medians = [np.median(d) for d in data_by_stage]
    ax.plot(range(n_stages), medians, 'k--', lw=1.5, alpha=0.6, zorder=0)
    
    # Calculate and display tau
    tau, p = kendalltau(pt_valid, idx_valid)
    sig_str = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    
    ax.text(0.97, 0.97, f"τ = {tau:.2f}{sig_str}", transform=ax.transAxes,
            ha='right', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                     edgecolor='#4CAF50', lw=1.5))
    
    ax.set_title("B. Pseudotime Recovers Stage Ordering",
                 fontweight='bold', loc='left', fontsize=12)
    
    # Interpretation text
    ax.text(0.5, -0.12, "Pseudotime increases monotonically with disease stage",
            transform=ax.transAxes, ha='center', fontsize=10,
            style='italic', color='#555555')


def main():
    """Create the two-panel manifold figure."""
    logger.info(f"Loading data for {TARGET_MODEL} on {TARGET_PROGRESSION}...")
    
    try:
        dcs, centroids, stages, pseudotime, classes = get_real_data()
        logger.info(f"Loaded {len(pseudotime)} samples across {len(stages)} stages")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return
    
    # Project to 2D
    X_2d, centroids_2d = rotate_to_progression_axis(dcs, centroids, stages)
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    
    ax_manifold = fig.add_subplot(gs[0])
    ax_violin = fig.add_subplot(gs[1])
    
    # Draw panels
    draw_manifold_panel(ax_manifold, X_2d, centroids_2d, stages, pseudotime, classes)
    draw_pseudotime_panel(ax_violin, stages, pseudotime, classes)
    
    plt.tight_layout()
    
    # Save
    save_path = OUTPUT_DIR / f'Figure_Manifold_{TARGET_PROGRESSION}_{TARGET_MODEL}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    logger.info(f"Saved figure to {save_path}")


if __name__ == "__main__":
    main()