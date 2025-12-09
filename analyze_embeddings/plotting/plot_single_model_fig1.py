#!/usr/bin/env python
"""
Method Overview Plotter - Improved Version
Design: Clean left-to-right pipeline with side-by-side comparison plots
Key features:
- 5 patches in 2 rows (3 top, 2 bottom) - unordered to show variety
- Side-by-side scatter plots for direct comparison (PCA vs Diffusion Manifold)
- Gradient trajectory line for visual continuity
- Cleaner annotations with better visual hierarchy
- Prominent tau score badge as pipeline output
- Diffusion plot uses DC1 and DC2 (skipping trivial DC0)
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from PIL import Image
import fsspec
import scanpy as sc
from scipy.stats import kendalltau

# --- Project Imports ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config
import plotting.plot_config as pcfg
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import (
    DPTConfig,
    cohort_to_anndata,
    compute_dpt,
)

# --- Configuration ---
PROGRESSION = "BDC"
MODEL_FOCUS = "uni2"
EMBEDDING_TYPE = "final_embedding"
N_PATCHES_DISPLAY = 5  # Back to 5 for better manifold illustration

# Color scheme
COLORS = {
    'early': '#2166AC',      # Blue
    'mid': '#92C5DE',        # Light blue  
    'late': '#B2182B',       # Red
    'arrow': '#BDBDBD',      # Gray for arrows
    'text_dark': '#2D2D2D',  # Dark gray for titles
    'text_light': '#666666', # Light gray for subtitles
    'grid': '#E8E8E8',       # Very light gray for grid
    'accent': '#1976D2',     # Blue accent for model box
}

# Create a blue-to-red colormap for progression
PROGRESSION_CMAP = LinearSegmentedColormap.from_list(
    'progression', [COLORS['early'], COLORS['mid'], COLORS['late']]
)

SHORT_NAMES = {
    "Ductal carcinoma in situ (low-grade)": "DCIS\n(Low Grade)",
    "Ductal carcinoma in situ (high-grade)": "DCIS\n(High Grade)",
    "Invasive non-special type carcinoma": "Invasive\nCarcinoma",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Data Loading Functions (unchanged from original) ---
def get_model_data(prog_name, model_name, patch_ids, prog_config):
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"], prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"], models=[model_name],
        progression_name=prog_name, scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    dataset.load_model_into_memory(model_name, patch_ids, EMBEDDING_TYPE)
    cohort_df = dataset.get_cohort(patch_ids, model_name, EMBEDDING_TYPE)
    if cohort_df.empty:
        raise ValueError(f"No data for {model_name}")

    adata = cohort_to_anndata(cohort_df, prog_config["classes"])
    sc.pp.pca(adata, n_comps=2)
    
    dpt_conf = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=max(10, config.DPT["n_diffusion_components"]), 
        metrics=set(), subsample_size=2000
    )
    compute_dpt(adata, prog_config["root_class"], dpt_conf)
    
    class_map = {c: i for i, c in enumerate(prog_config["classes"])}
    adata.obs['class_int'] = adata.obs['class'].map(class_map)
    valid = adata.obs[np.isfinite(adata.obs['dpt_pseudotime'])]
    tau, _ = kendalltau(valid['class_int'], valid['dpt_pseudotime'])
    adata.uns['tau'] = tau
    return adata, cohort_df


def select_patches_by_dpt(adata, n_samples=3):
    """Select patches evenly spaced along the DPT trajectory."""
    if 'dpt_pseudotime' not in adata.obs:
        raise ValueError("DPT missing.")
    df = adata.obs.copy()
    df = df[np.isfinite(df['dpt_pseudotime'])].sort_values('dpt_pseudotime')
    target_values = np.linspace(df['dpt_pseudotime'].min(), df['dpt_pseudotime'].max(), n_samples)
    selected_indices = []
    dpt_values = df['dpt_pseudotime'].values
    for t in target_values:
        idx_loc = (np.abs(dpt_values - t)).argmin()
        selected_indices.append(df.iloc[idx_loc].name)
    selected_df = df.loc[selected_indices]
    results = []
    for idx, row in selected_df.iterrows():
        results.append({
            'slide_id': row['slide_id'], 'patch_id': row['patch_id'],
            'class': row['class'], 'dpt_pseudotime': row['dpt_pseudotime']
        })
    return results


def fetch_images_from_selection(selection_list, prog_config):
    results = []
    fs = fsspec.filesystem("gcs")
    image_subdir = prog_config.get("image_subdir", "imagenet")
    for item in selection_list:
        slide_id = str(item['slide_id'])
        if not slide_id.startswith("slide_"):
            slide_id_path = f"slide_{slide_id}"
        else:
            slide_id_path = slide_id
        patch_id = str(item['patch_id'])
        if not patch_id.startswith("patch_"):
            patch_id_path = f"patch_{patch_id}"
        else:
            patch_id_path = patch_id
        
        gcs_path = f"gs://{prog_config['bucket']}/{image_subdir}/train/{item['class']}/{slide_id_path}/{patch_id_path}.png"
        short_class = SHORT_NAMES.get(item['class'], item['class'])
        try:
            with fs.open(gcs_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            results.append((img, short_class, item['slide_id'], item['patch_id']))
        except Exception as e:
            logger.warning(f"Could not load image: {gcs_path}")
            results.append((None, "Missing", item['slide_id'], item['patch_id']))
    return results


def find_loc(adata, slide_id, patch_id):
    obs = adata.obs
    mask = (obs['slide_id'].astype(str) == str(slide_id)) & \
           (obs['patch_id'].astype(str) == str(patch_id))
    hits = np.where(mask)[0]
    if len(hits) == 0:
        return None
    return hits[0]


# --- Drawing Helpers ---
def draw_gradient_line(ax, points, cmap, linewidth=3, alpha=0.9):
    """Draw a line with gradient color along its length."""
    points = np.array(points)
    segments = np.array([[points[i], points[i+1]] for i in range(len(points)-1)])
    
    # Create normalized color values
    norm = plt.Normalize(0, len(segments))
    colors = [cmap(norm(i + 0.5)) for i in range(len(segments))]
    
    lc = LineCollection(segments, colors=colors, linewidth=linewidth, alpha=alpha, zorder=5)
    ax.add_collection(lc)
    return lc


def create_model_box(fig, center_x, center_y, width=0.08, height=0.12):
    """Create a clean model box icon."""
    # Main box
    box = mpatches.FancyBboxPatch(
        (center_x - width/2, center_y - height/2), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        ec=COLORS['accent'], fc='white', lw=2,
        transform=fig.transFigure, zorder=10
    )
    fig.add_artist(box)
    
    # Add "layers" visual inside box
    layer_width = width * 0.6
    layer_height = height * 0.12
    layer_gap = height * 0.08
    start_y = center_y + height * 0.25
    
    for i in range(3):
        layer = mpatches.FancyBboxPatch(
            (center_x - layer_width/2, start_y - i * (layer_height + layer_gap)),
            layer_width, layer_height,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            ec='none', fc=COLORS['accent'], alpha=0.3 + i * 0.2,
            transform=fig.transFigure, zorder=11
        )
        fig.add_artist(layer)
    
    # Label
    fig.text(center_x, center_y - height/2 - 0.025, "Foundation\nModel",
             ha='center', va='top', fontsize=9, fontweight='medium',
             color=COLORS['text_dark'], transform=fig.transFigure)


def plot_embedding_space(ax, adata, basis, locs, prog_config, show_trajectory=False, title=""):
    """Plot embedding space with consistent styling."""
    coords = adata.obsm[basis]
    if basis == 'X_diffmap':
        # Skip DC0 (trivial/constant component), use DC1 and DC2
        coords = coords[:, 1:3]
    else:
        coords = coords[:, :2]
    
    classes = prog_config["classes"]
    n_classes = len(classes)
    
    # Plot all points with class-based colors
    for i, cls in enumerate(classes):
        mask = adata.obs['class'] == cls
        color = PROGRESSION_CMAP(i / (n_classes - 1))
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                   c=[color], s=15, alpha=0.25, edgecolors='none', rasterized=True)
    
    # Get trajectory coordinates
    path_coords = coords[locs]
    
    # Draw trajectory line with gradient (only for diffusion plot)
    if show_trajectory:
        draw_gradient_line(ax, path_coords, PROGRESSION_CMAP, linewidth=3.5, alpha=0.95)
    
    # Plot trajectory points with consistent circle markers
    for i, (x, y) in enumerate(path_coords):
        color = PROGRESSION_CMAP(i / (len(locs) - 1))
        ax.scatter(x, y, c=[color], s=180, edgecolors='white', linewidth=2, zorder=15)
        ax.scatter(x, y, c=[color], s=180, edgecolors=COLORS['text_dark'], 
                   linewidth=1, zorder=15, alpha=0.8)
    
    # Clean styling
    ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['text_dark'], pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Subtle box
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1)
    
    ax.set_facecolor('white')


def main():
    # 1. Load Data
    prog_config = next(p for p in config.PROGRESSIONS if p["name"] == PROGRESSION)
    temp_reg = RegistryConfig(
        bucket=prog_config["bucket"], prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"], models=[MODEL_FOCUS], 
        progression_name=PROGRESSION, scan_all_models=False
    )
    temp_ds = ProgressionEmbeddingDataset(temp_reg)
    patch_ids = temp_ds.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=config.EVALUATION["seed"]
    )
    
    logger.info(f"Loading data for {MODEL_FOCUS}...")
    adata, df = get_model_data(PROGRESSION, MODEL_FOCUS, patch_ids, prog_config)
    logger.info(f"Selecting {N_PATCHES_DISPLAY} patches based on DPT...")
    selected_patches_info = select_patches_by_dpt(adata, n_samples=N_PATCHES_DISPLAY)
    logger.info("Fetching images...")
    image_data = fetch_images_from_selection(selected_patches_info, prog_config)
    
    locs = [find_loc(adata, x[2], x[3]) for x in image_data]
    valid_indices = [i for i, l in enumerate(locs) if l is not None]
    if len(valid_indices) < len(image_data):
        image_data = [image_data[i] for i in valid_indices]
        locs = [locs[i] for i in valid_indices]
    tau_score = adata.uns['tau']

    # 2. Create Figure
    fig = plt.figure(figsize=(14, 5.5), facecolor='white')
    
    # --- Layout Constants ---
    # Patches section: left side, 2 rows (3 top, 2 bottom)
    patch_left = 0.03
    patch_width = 0.05
    patch_height = 0.18
    patch_gap_x = 0.015
    patch_gap_y = 0.06
    patch_top_y = 0.52
    patch_bottom_y = patch_top_y - patch_height - patch_gap_y
    
    # Model box
    model_x = 0.34
    model_y = (patch_top_y + patch_bottom_y + patch_height) / 2  # Center between rows
    
    # Scatter plots: right side (side by side)
    plot_left = 0.46
    plot_width = 0.22
    plot_height = 0.65
    plot_gap = 0.06
    plot_bottom = 0.18
    
    # --- Title ---
    fig.text(0.03, 0.94, "Measuring Trajectory Fidelity in Foundation Model Representations",
             fontsize=14, fontweight='bold', color=COLORS['text_dark'])
    fig.text(0.03, 0.88, f"Disease progression: Breast Ductal Carcinoma  •  Model: {pcfg.MODEL_LABELS.get(MODEL_FOCUS, MODEL_FOCUS)}",
             fontsize=10, color=COLORS['text_light'])
    
    # --- Input Patches (2 rows: 3 on top, 2 on bottom) ---
    fig.text(patch_left, patch_top_y + patch_height + 0.04, "Input Patches",
             fontsize=10, fontweight='bold', color=COLORS['text_dark'])
    fig.text(patch_left + 0.08, patch_top_y + patch_height + 0.04, "(unordered samples)",
             fontsize=8, color=COLORS['text_light'])
    
    # Arrange patches: top row gets indices 0, 2, 4; bottom row gets 1, 3
    top_row_indices = [0, 2, 4]
    bottom_row_indices = [1, 3]
    
    # Center the bottom row
    bottom_row_offset = (len(top_row_indices) - len(bottom_row_indices)) * (patch_width + patch_gap_x) / 2
    
    for i, (img, label, _, _) in enumerate(image_data):
        if i in top_row_indices:
            row_idx = top_row_indices.index(i)
            ax_x = patch_left + row_idx * (patch_width + patch_gap_x)
            ax_y = patch_top_y
        else:
            row_idx = bottom_row_indices.index(i)
            ax_x = patch_left + bottom_row_offset + row_idx * (patch_width + patch_gap_x)
            ax_y = patch_bottom_y
        
        ax = fig.add_axes([ax_x, ax_y, patch_width, patch_height])
        
        if img:
            ax.imshow(img)
            # Color border by progression stage
            border_color = PROGRESSION_CMAP(i / (len(image_data) - 1))
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # --- Arrow: Patches to Model ---
    arrow_y = model_y
    patches_end = patch_left + len(top_row_indices) * (patch_width + patch_gap_x) - patch_gap_x + 0.025
    
    fig.add_artist(mpatches.FancyArrowPatch(
        (patches_end, arrow_y), (model_x - 0.05, arrow_y),
        arrowstyle='-|>,head_length=6,head_width=4',
        color=COLORS['arrow'], lw=2, transform=fig.transFigure, zorder=1
    ))
    
    # --- Model Box ---
    create_model_box(fig, model_x, model_y)
    
    # --- Arrow: Model to Plots ---
    fig.add_artist(mpatches.FancyArrowPatch(
        (model_x + 0.055, arrow_y), (plot_left - 0.03, arrow_y),
        arrowstyle='-|>,head_length=6,head_width=4',
        color=COLORS['arrow'], lw=2, transform=fig.transFigure, zorder=1
    ))
    
    # --- Scatter Plots (Side by Side) ---
    # Left plot: Raw PCA
    ax_pca = fig.add_axes([plot_left, plot_bottom, plot_width, plot_height])
    plot_embedding_space(ax_pca, adata, 'X_pca', locs, prog_config,
                         show_trajectory=False, title="Raw Embedding (PCA)")
    
    # Arrow between plots
    arrow_mid_y = plot_bottom + plot_height / 2
    fig.add_artist(mpatches.FancyArrowPatch(
        (plot_left + plot_width + 0.008, arrow_mid_y),
        (plot_left + plot_width + plot_gap - 0.008, arrow_mid_y),
        arrowstyle='-|>,head_length=5,head_width=3',
        color=COLORS['arrow'], lw=1.5, transform=fig.transFigure
    ))
    fig.text(plot_left + plot_width + plot_gap/2, arrow_mid_y + 0.04, "DPT",
             ha='center', fontsize=8, color=COLORS['text_light'], fontstyle='italic')
    
    # Right plot: Diffusion Manifold
    diff_plot_left = plot_left + plot_width + plot_gap
    ax_diff = fig.add_axes([diff_plot_left, plot_bottom, plot_width, plot_height])
    plot_embedding_space(ax_diff, adata, 'X_diffmap', locs, prog_config,
                         show_trajectory=True, title="Diffusion Manifold")
    
    # --- Tau Score Badge (positioned inside the diffusion plot, top-right) ---
    # Place it as an inset in the upper right corner of the diffusion plot
    tau_x = diff_plot_left + plot_width - 0.065
    tau_y = plot_bottom + plot_height - 0.12
    
    # Badge background
    badge = mpatches.FancyBboxPatch(
        (tau_x, tau_y), 0.055, 0.10,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        ec=COLORS['accent'], fc='white', lw=1.5, alpha=0.95,
        transform=fig.transFigure, zorder=20
    )
    fig.add_artist(badge)
    
    fig.text(tau_x + 0.0275, tau_y + 0.075, "τ",
             ha='center', va='center', fontsize=11, fontweight='bold',
             color=COLORS['accent'], transform=fig.transFigure, zorder=21)
    fig.text(tau_x + 0.0275, tau_y + 0.035, f"{tau_score:.2f}",
             ha='center', va='center', fontsize=13, fontweight='bold',
             color=COLORS['text_dark'], transform=fig.transFigure, zorder=21)
    
    # Label explaining tau metric
    fig.text(tau_x + 0.0275, tau_y - 0.02, "Kendall's τ",
             ha='center', va='top', fontsize=6.5, color=COLORS['text_light'],
             transform=fig.transFigure, zorder=21)
    fig.text(tau_x + 0.0275, tau_y - 0.04, "(pseudotime vs. stage)",
             ha='center', va='top', fontsize=5.5, color=COLORS['text_light'],
             transform=fig.transFigure, zorder=21)
    
    # --- Color Legend (centered below plots) ---
    legend_x = plot_left + plot_width / 2
    legend_y = plot_bottom - 0.08
    
    # Gradient bar
    gradient_width = 0.12
    gradient_height = 0.012
    gradient_ax = fig.add_axes([legend_x, legend_y, gradient_width, gradient_height])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient_ax.imshow(gradient, aspect='auto', cmap=PROGRESSION_CMAP)
    gradient_ax.set_xticks([])
    gradient_ax.set_yticks([])
    for spine in gradient_ax.spines.values():
        spine.set_visible(False)
    
    fig.text(legend_x - 0.005, legend_y + gradient_height/2, "Early", fontsize=7, ha='right',
             va='center', color=COLORS['text_light'])
    fig.text(legend_x + gradient_width + 0.005, legend_y + gradient_height/2, "Late", fontsize=7, ha='left',
             va='center', color=COLORS['text_light'])
    fig.text(legend_x + gradient_width/2, legend_y - 0.025, "Disease Stage", fontsize=7, 
             ha='center', color=COLORS['text_dark'], fontweight='medium')
    
    # --- Save ---
    output_path = config.PLOTS_OUTPUT_DIR / f"method_overview_clean_{PROGRESSION}_{MODEL_FOCUS}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ Saved to {output_path}")
    
    # Also save a PDF version for publication
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ PDF saved to {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
# #!/usr/bin/env python
# """
# Method Overview Plotter - Improved Version
# Design: Clean left-to-right pipeline with side-by-side comparison plots
# Key features:
# - 5 patches in 2 rows (3 top, 2 bottom) - unordered to show variety
# - Side-by-side scatter plots for direct comparison (PCA vs Diffusion Manifold)
# - Gradient trajectory line for visual continuity
# - Cleaner annotations with better visual hierarchy
# - Prominent tau score badge as pipeline output
# - Diffusion plot uses DC1 and DC2 (skipping trivial DC0)
# """

# import sys
# import logging
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.collections import LineCollection
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.gridspec import GridSpec
# from PIL import Image
# import fsspec
# import scanpy as sc
# from scipy.stats import kendalltau

# # --- Project Imports ---
# project_root = Path(__file__).resolve().parent.parent
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

# import config
# import plotting.plot_config as pcfg
# from data.progression_embedding_dataset import (
#     ProgressionEmbeddingDataset,
#     RegistryConfig,
# )
# from analysis.dpt import (
#     DPTConfig,
#     cohort_to_anndata,
#     compute_dpt,
# )

# # --- Configuration ---
# PROGRESSION = "BDC"
# MODEL_FOCUS = "uni2"
# EMBEDDING_TYPE = "final_embedding"
# N_PATCHES_DISPLAY = 5  # Back to 5 for better manifold illustration

# # Color scheme
# COLORS = {
#     'early': '#2166AC',      # Blue
#     'mid': '#92C5DE',        # Light blue  
#     'late': '#B2182B',       # Red
#     'arrow': '#BDBDBD',      # Gray for arrows
#     'text_dark': '#2D2D2D',  # Dark gray for titles
#     'text_light': '#666666', # Light gray for subtitles
#     'grid': '#E8E8E8',       # Very light gray for grid
#     'accent': '#1976D2',     # Blue accent for model box
# }

# # Create a blue-to-red colormap for progression
# PROGRESSION_CMAP = LinearSegmentedColormap.from_list(
#     'progression', [COLORS['early'], COLORS['mid'], COLORS['late']]
# )

# SHORT_NAMES = {
#     "Ductal carcinoma in situ (low-grade)": "DCIS\n(Low Grade)",
#     "Ductal carcinoma in situ (high-grade)": "DCIS\n(High Grade)",
#     "Invasive non-special type carcinoma": "Invasive\nCarcinoma",
# }

# logging.basicConfig(level=logging.INFO, format="%(message)s")
# logger = logging.getLogger(__name__)


# # --- Data Loading Functions (unchanged from original) ---
# def get_model_data(prog_name, model_name, patch_ids, prog_config):
#     registry_config = RegistryConfig(
#         bucket=prog_config["bucket"], prefix=prog_config["prefix"],
#         ordered_classes=prog_config["classes"], models=[model_name],
#         progression_name=prog_name, scan_all_models=False
#     )
#     dataset = ProgressionEmbeddingDataset(registry_config)
#     dataset.load_model_into_memory(model_name, patch_ids, EMBEDDING_TYPE)
#     cohort_df = dataset.get_cohort(patch_ids, model_name, EMBEDDING_TYPE)
#     if cohort_df.empty:
#         raise ValueError(f"No data for {model_name}")

#     adata = cohort_to_anndata(cohort_df, prog_config["classes"])
#     sc.pp.pca(adata, n_comps=2)
    
#     dpt_conf = DPTConfig(
#         n_neighbors=config.DPT["n_neighbors"],
#         n_diffusion_components=max(10, config.DPT["n_diffusion_components"]), 
#         metrics=set(), subsample_size=2000
#     )
#     compute_dpt(adata, prog_config["root_class"], dpt_conf)
    
#     class_map = {c: i for i, c in enumerate(prog_config["classes"])}
#     adata.obs['class_int'] = adata.obs['class'].map(class_map)
#     valid = adata.obs[np.isfinite(adata.obs['dpt_pseudotime'])]
#     tau, _ = kendalltau(valid['class_int'], valid['dpt_pseudotime'])
#     adata.uns['tau'] = tau
#     return adata, cohort_df


# def select_patches_by_dpt(adata, n_samples=3):
#     """Select patches evenly spaced along the DPT trajectory."""
#     if 'dpt_pseudotime' not in adata.obs:
#         raise ValueError("DPT missing.")
#     df = adata.obs.copy()
#     df = df[np.isfinite(df['dpt_pseudotime'])].sort_values('dpt_pseudotime')
#     target_values = np.linspace(df['dpt_pseudotime'].min(), df['dpt_pseudotime'].max(), n_samples)
#     selected_indices = []
#     dpt_values = df['dpt_pseudotime'].values
#     for t in target_values:
#         idx_loc = (np.abs(dpt_values - t)).argmin()
#         selected_indices.append(df.iloc[idx_loc].name)
#     selected_df = df.loc[selected_indices]
#     results = []
#     for idx, row in selected_df.iterrows():
#         results.append({
#             'slide_id': row['slide_id'], 'patch_id': row['patch_id'],
#             'class': row['class'], 'dpt_pseudotime': row['dpt_pseudotime']
#         })
#     return results


# def fetch_images_from_selection(selection_list, prog_config):
#     results = []
#     fs = fsspec.filesystem("gcs")
#     image_subdir = prog_config.get("image_subdir", "imagenet")
#     for item in selection_list:
#         slide_id = str(item['slide_id'])
#         if not slide_id.startswith("slide_"):
#             slide_id_path = f"slide_{slide_id}"
#         else:
#             slide_id_path = slide_id
#         patch_id = str(item['patch_id'])
#         if not patch_id.startswith("patch_"):
#             patch_id_path = f"patch_{patch_id}"
#         else:
#             patch_id_path = patch_id
        
#         gcs_path = f"gs://{prog_config['bucket']}/{image_subdir}/train/{item['class']}/{slide_id_path}/{patch_id_path}.png"
#         short_class = SHORT_NAMES.get(item['class'], item['class'])
#         try:
#             with fs.open(gcs_path, 'rb') as f:
#                 img = Image.open(f).convert("RGB")
#             results.append((img, short_class, item['slide_id'], item['patch_id']))
#         except Exception as e:
#             logger.warning(f"Could not load image: {gcs_path}")
#             results.append((None, "Missing", item['slide_id'], item['patch_id']))
#     return results


# def find_loc(adata, slide_id, patch_id):
#     obs = adata.obs
#     mask = (obs['slide_id'].astype(str) == str(slide_id)) & \
#            (obs['patch_id'].astype(str) == str(patch_id))
#     hits = np.where(mask)[0]
#     if len(hits) == 0:
#         return None
#     return hits[0]


# # --- Drawing Helpers ---
# def draw_gradient_line(ax, points, cmap, linewidth=3, alpha=0.9):
#     """Draw a line with gradient color along its length."""
#     points = np.array(points)
#     segments = np.array([[points[i], points[i+1]] for i in range(len(points)-1)])
    
#     # Create normalized color values
#     norm = plt.Normalize(0, len(segments))
#     colors = [cmap(norm(i + 0.5)) for i in range(len(segments))]
    
#     lc = LineCollection(segments, colors=colors, linewidth=linewidth, alpha=alpha, zorder=5)
#     ax.add_collection(lc)
#     return lc


# def create_model_box(fig, center_x, center_y, width=0.08, height=0.12):
#     """Create a clean model box icon."""
#     # Main box
#     box = mpatches.FancyBboxPatch(
#         (center_x - width/2, center_y - height/2), width, height,
#         boxstyle="round,pad=0.01,rounding_size=0.02",
#         ec=COLORS['accent'], fc='white', lw=2,
#         transform=fig.transFigure, zorder=10
#     )
#     fig.add_artist(box)
    
#     # Add "layers" visual inside box
#     layer_width = width * 0.6
#     layer_height = height * 0.12
#     layer_gap = height * 0.08
#     start_y = center_y + height * 0.25
    
#     for i in range(3):
#         layer = mpatches.FancyBboxPatch(
#             (center_x - layer_width/2, start_y - i * (layer_height + layer_gap)),
#             layer_width, layer_height,
#             boxstyle="round,pad=0.002,rounding_size=0.01",
#             ec='none', fc=COLORS['accent'], alpha=0.3 + i * 0.2,
#             transform=fig.transFigure, zorder=11
#         )
#         fig.add_artist(layer)
    
#     # Label
#     fig.text(center_x, center_y - height/2 - 0.025, "Foundation\nModel",
#              ha='center', va='top', fontsize=9, fontweight='medium',
#              color=COLORS['text_dark'], transform=fig.transFigure)


# def plot_embedding_space(ax, adata, basis, locs, prog_config, show_trajectory=False, title=""):
#     """Plot embedding space with consistent styling."""
#     coords = adata.obsm[basis]
#     if basis == 'X_diffmap':
#         # Skip DC0 (trivial/constant component), use DC1 and DC2
#         coords = coords[:, 1:3]
#     else:
#         coords = coords[:, :2]
    
#     classes = prog_config["classes"]
#     n_classes = len(classes)
    
#     # Plot all points with class-based colors
#     for i, cls in enumerate(classes):
#         mask = adata.obs['class'] == cls
#         color = PROGRESSION_CMAP(i / (n_classes - 1))
#         ax.scatter(coords[mask, 0], coords[mask, 1], 
#                    c=[color], s=15, alpha=0.25, edgecolors='none', rasterized=True)
    
#     # Get trajectory coordinates
#     path_coords = coords[locs]
    
#     # Draw trajectory line with gradient (only for diffusion plot)
#     if show_trajectory:
#         draw_gradient_line(ax, path_coords, PROGRESSION_CMAP, linewidth=3.5, alpha=0.95)
    
#     # Plot trajectory points with consistent circle markers
#     for i, (x, y) in enumerate(path_coords):
#         color = PROGRESSION_CMAP(i / (len(locs) - 1))
#         ax.scatter(x, y, c=[color], s=180, edgecolors='white', linewidth=2, zorder=15)
#         ax.scatter(x, y, c=[color], s=180, edgecolors=COLORS['text_dark'], 
#                    linewidth=1, zorder=15, alpha=0.8)
    
#     # Clean styling
#     ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['text_dark'], pad=10)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     # Subtle box
#     for spine in ax.spines.values():
#         spine.set_color(COLORS['grid'])
#         spine.set_linewidth(1)
    
#     ax.set_facecolor('white')


# def main():
#     # 1. Load Data
#     prog_config = next(p for p in config.PROGRESSIONS if p["name"] == PROGRESSION)
#     temp_reg = RegistryConfig(
#         bucket=prog_config["bucket"], prefix=prog_config["prefix"],
#         ordered_classes=prog_config["classes"], models=[MODEL_FOCUS], 
#         progression_name=PROGRESSION, scan_all_models=False
#     )
#     temp_ds = ProgressionEmbeddingDataset(temp_reg)
#     patch_ids = temp_ds.sample_patch_ids(
#         n_per_class=config.EVALUATION["n_per_class"],
#         max_per_slide=config.EVALUATION["max_per_slide"],
#         seed=config.EVALUATION["seed"]
#     )
    
#     logger.info(f"Loading data for {MODEL_FOCUS}...")
#     adata, df = get_model_data(PROGRESSION, MODEL_FOCUS, patch_ids, prog_config)
#     logger.info(f"Selecting {N_PATCHES_DISPLAY} patches based on DPT...")
#     selected_patches_info = select_patches_by_dpt(adata, n_samples=N_PATCHES_DISPLAY)
#     logger.info("Fetching images...")
#     image_data = fetch_images_from_selection(selected_patches_info, prog_config)
    
#     locs = [find_loc(adata, x[2], x[3]) for x in image_data]
#     valid_indices = [i for i, l in enumerate(locs) if l is not None]
#     if len(valid_indices) < len(image_data):
#         image_data = [image_data[i] for i in valid_indices]
#         locs = [locs[i] for i in valid_indices]
#     tau_score = adata.uns['tau']

#     # 2. Create Figure
#     fig = plt.figure(figsize=(14, 5.5), facecolor='white')
    
#     # --- Layout Constants ---
#     # Patches section: left side, 2 rows (3 top, 2 bottom)
#     patch_left = 0.03
#     patch_width = 0.05
#     patch_height = 0.18
#     patch_gap_x = 0.015
#     patch_gap_y = 0.06
#     patch_top_y = 0.52
#     patch_bottom_y = patch_top_y - patch_height - patch_gap_y
    
#     # Model box
#     model_x = 0.34
#     model_y = (patch_top_y + patch_bottom_y + patch_height) / 2  # Center between rows
    
#     # Scatter plots: right side (side by side)
#     plot_left = 0.46
#     plot_width = 0.22
#     plot_height = 0.65
#     plot_gap = 0.06
#     plot_bottom = 0.18
    
#     # --- Title ---
#     fig.text(0.03, 0.94, "Measuring Trajectory Fidelity in Foundation Model Representations",
#              fontsize=14, fontweight='bold', color=COLORS['text_dark'])
#     fig.text(0.03, 0.88, f"Disease progression: Breast Ductal Carcinoma  •  Model: {pcfg.MODEL_LABELS.get(MODEL_FOCUS, MODEL_FOCUS)}",
#              fontsize=10, color=COLORS['text_light'])
    
#     # --- Input Patches (2 rows: 3 on top, 2 on bottom) ---
#     fig.text(patch_left, patch_top_y + patch_height + 0.04, "Input Patches",
#              fontsize=10, fontweight='bold', color=COLORS['text_dark'])
#     fig.text(patch_left + 0.08, patch_top_y + patch_height + 0.04, "(unordered samples)",
#              fontsize=8, color=COLORS['text_light'])
    
#     # Arrange patches: top row gets indices 0, 2, 4; bottom row gets 1, 3
#     top_row_indices = [0, 2, 4]
#     bottom_row_indices = [1, 3]
    
#     # Center the bottom row
#     bottom_row_offset = (len(top_row_indices) - len(bottom_row_indices)) * (patch_width + patch_gap_x) / 2
    
#     for i, (img, label, _, _) in enumerate(image_data):
#         if i in top_row_indices:
#             row_idx = top_row_indices.index(i)
#             ax_x = patch_left + row_idx * (patch_width + patch_gap_x)
#             ax_y = patch_top_y
#         else:
#             row_idx = bottom_row_indices.index(i)
#             ax_x = patch_left + bottom_row_offset + row_idx * (patch_width + patch_gap_x)
#             ax_y = patch_bottom_y
        
#         ax = fig.add_axes([ax_x, ax_y, patch_width, patch_height])
        
#         if img:
#             ax.imshow(img)
#             # Color border by progression stage
#             border_color = PROGRESSION_CMAP(i / (len(image_data) - 1))
#             for spine in ax.spines.values():
#                 spine.set_edgecolor(border_color)
#                 spine.set_linewidth(2.5)
#         ax.set_xticks([])
#         ax.set_yticks([])
        
#         # Stage label below each patch
#         fig.text(ax_x + patch_width/2, ax_y - 0.025, label,
#                  ha='center', va='top', fontsize=6, color=COLORS['text_dark'],
#                  linespacing=0.85)
    
#     # --- Arrow: Patches to Model ---
#     arrow_y = model_y
#     patches_end = patch_left + len(top_row_indices) * (patch_width + patch_gap_x) - patch_gap_x + 0.015
    
#     fig.add_artist(mpatches.FancyArrowPatch(
#         (patches_end, arrow_y), (model_x - 0.045, arrow_y),
#         arrowstyle='-|>,head_length=6,head_width=4',
#         color=COLORS['arrow'], lw=2, transform=fig.transFigure, zorder=1
#     ))
    
#     # --- Model Box ---
#     create_model_box(fig, model_x, model_y)
    
#     # --- Arrow: Model to Plots ---
#     fig.add_artist(mpatches.FancyArrowPatch(
#         (model_x + 0.055, arrow_y), (plot_left - 0.03, arrow_y),
#         arrowstyle='-|>,head_length=6,head_width=4',
#         color=COLORS['arrow'], lw=2, transform=fig.transFigure, zorder=1
#     ))
#     fig.text((model_x + 0.055 + plot_left - 0.03) / 2, arrow_y + 0.035, "embed",
#              ha='center', fontsize=8, color=COLORS['text_light'], fontstyle='italic')
    
#     # --- Scatter Plots (Side by Side) ---
#     # Left plot: Raw PCA
#     ax_pca = fig.add_axes([plot_left, plot_bottom, plot_width, plot_height])
#     plot_embedding_space(ax_pca, adata, 'X_pca', locs, prog_config,
#                          show_trajectory=False, title="Raw Embedding (PCA)")
    
#     # Arrow between plots
#     arrow_mid_y = plot_bottom + plot_height / 2
#     fig.add_artist(mpatches.FancyArrowPatch(
#         (plot_left + plot_width + 0.008, arrow_mid_y),
#         (plot_left + plot_width + plot_gap - 0.008, arrow_mid_y),
#         arrowstyle='-|>,head_length=5,head_width=3',
#         color=COLORS['arrow'], lw=1.5, transform=fig.transFigure
#     ))
#     fig.text(plot_left + plot_width + plot_gap/2, arrow_mid_y + 0.04, "DPT",
#              ha='center', fontsize=8, color=COLORS['text_light'], fontstyle='italic')
    
#     # Right plot: Diffusion Manifold
#     diff_plot_left = plot_left + plot_width + plot_gap
#     ax_diff = fig.add_axes([diff_plot_left, plot_bottom, plot_width, plot_height])
#     plot_embedding_space(ax_diff, adata, 'X_diffmap', locs, prog_config,
#                          show_trajectory=True, title="Diffusion Manifold")
    
#     # --- Tau Score Badge (positioned inside the diffusion plot, top-right) ---
#     # Place it as an inset in the upper right corner of the diffusion plot
#     tau_x = diff_plot_left + plot_width - 0.065
#     tau_y = plot_bottom + plot_height - 0.12
    
#     # Badge background
#     badge = mpatches.FancyBboxPatch(
#         (tau_x, tau_y), 0.055, 0.10,
#         boxstyle="round,pad=0.008,rounding_size=0.015",
#         ec=COLORS['accent'], fc='white', lw=1.5, alpha=0.95,
#         transform=fig.transFigure, zorder=20
#     )
#     fig.add_artist(badge)
    
#     fig.text(tau_x + 0.0275, tau_y + 0.075, "τ",
#              ha='center', va='center', fontsize=11, fontweight='bold',
#              color=COLORS['accent'], transform=fig.transFigure, zorder=21)
#     fig.text(tau_x + 0.0275, tau_y + 0.035, f"{tau_score:.2f}",
#              ha='center', va='center', fontsize=13, fontweight='bold',
#              color=COLORS['text_dark'], transform=fig.transFigure, zorder=21)
    
#     # --- Color Legend (centered below plots) ---
#     legend_x = plot_left + plot_width / 2
#     legend_y = plot_bottom - 0.08
    
#     # Gradient bar
#     gradient_width = 0.12
#     gradient_height = 0.012
#     gradient_ax = fig.add_axes([legend_x, legend_y, gradient_width, gradient_height])
#     gradient = np.linspace(0, 1, 256).reshape(1, -1)
#     gradient_ax.imshow(gradient, aspect='auto', cmap=PROGRESSION_CMAP)
#     gradient_ax.set_xticks([])
#     gradient_ax.set_yticks([])
#     for spine in gradient_ax.spines.values():
#         spine.set_visible(False)
    
#     fig.text(legend_x - 0.005, legend_y + gradient_height/2, "Early", fontsize=7, ha='right',
#              va='center', color=COLORS['text_light'])
#     fig.text(legend_x + gradient_width + 0.005, legend_y + gradient_height/2, "Late", fontsize=7, ha='left',
#              va='center', color=COLORS['text_light'])
#     fig.text(legend_x + gradient_width/2, legend_y - 0.025, "Disease Stage", fontsize=7, 
#              ha='center', color=COLORS['text_dark'], fontweight='medium')
    
#     # --- Save ---
#     output_path = config.PLOTS_OUTPUT_DIR / f"method_overview_clean_{PROGRESSION}_{MODEL_FOCUS}.png"
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     logger.info(f"✅ Saved to {output_path}")
    
#     # Also save a PDF version for publication
#     pdf_path = output_path.with_suffix('.pdf')
#     plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
#     logger.info(f"✅ PDF saved to {pdf_path}")
    
#     plt.close()


# if __name__ == "__main__":
#     main()