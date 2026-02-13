#!/usr/bin/env python
"""
Method Overview Figure

Pipeline
    Input Patches → Foundation Model → DPT → [Manifold by Class] [Manifold by Pseudotime] → Tau Metric with Interpretation

Usage:
    python plotting/plot_method_overview_figure.py --progression SCC
    python plotting/plot_method_overview_figure.py --progression BDC --model uni2
"""

import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
import fsspec
import scanpy as sc
from scipy.stats import kendalltau
import warnings

# --- Project Imports ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from plotting import plot_config as pcfg
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
EMBEDDING_TYPE = "final_embedding"
N_PATCHES_DEFAULT = 5

# Color scheme
COLORS = {
    'early': '#2166AC',      # Blue (low pseudotime)
    'mid': '#F7F7F7',        # White (mid pseudotime)
    'late': '#B2182B',       # Red (high pseudotime)
    'arrow': '#9E9E9E',      # Gray for arrows
    'text_dark': '#2D2D2D',  # Dark gray for titles
    'text_light': '#666666', # Light gray for subtitles
    'grid': '#E0E0E0',       # Light gray for grid/borders
    'accent': '#1976D2',     # Blue accent
}

# Pseudotime colormap
PSEUDOTIME_CMAP = LinearSegmentedColormap.from_list(
    'pseudotime', [COLORS['early'], COLORS['mid'], COLORS['late']]
)

# Class/stage colors - distinct, colorblind-friendly
STAGE_COLORS = [
    '#E41A1C',  # Red
    '#377EB8',  # Blue  
    '#4DAF4A',  # Green
    '#984EA3',  # Purple
    '#FF7F00',  # Orange
    '#A65628',  # Brown
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Data Loading ---
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
    sc.pp.pca(adata, n_comps=50)
    
    dpt_conf = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=max(10, config.DPT["n_diffusion_components"]), 
        metrics=set(), subsample_size=2000
    )
    compute_dpt(adata, prog_config["root_class"], dpt_conf)
    
    class_map = {c: i for i, c in enumerate(prog_config["classes"])}
    adata.obs['class_int'] = adata.obs['class'].map(class_map)
    valid = adata.obs[np.isfinite(adata.obs['dpt_pseudotime'])]
    tau, p_value = kendalltau(valid['class_int'], valid['dpt_pseudotime'])
    adata.uns['tau'] = tau
    adata.uns['tau_pvalue'] = p_value
    return adata, cohort_df


def select_patches_by_dpt(adata, n_samples=5):
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

def select_patches_by_class(adata, prog_config, n_total=5):
    """
    Select patches that form a smooth trajectory along the manifold spine.
    
    Strategy:
    1. Find the "spine" of the manifold by binning pseudotime and taking
       the medoid (most central point) of each bin in DC1-DC2 space
    2. Select points along this spine that cover all classes
    """
    classes = prog_config["classes"]
    pseudotime = adata.obs['dpt_pseudotime']
    valid_mask = np.isfinite(pseudotime)
    coords = adata.obsm['X_diffmap'][:, 1:3]
    
    valid_indices = adata.obs[valid_mask].index
    valid_ilocs = np.array([adata.obs.index.get_loc(idx) for idx in valid_indices])
    valid_pts = pseudotime[valid_mask].values
    valid_coords = coords[valid_ilocs]
    valid_classes = adata.obs.loc[valid_indices, 'class'].values
    
    # Step 1: Find the manifold "spine" by computing medoids along pseudotime
    n_bins = 50
    pt_min, pt_max = valid_pts.min(), valid_pts.max()
    bin_edges = np.linspace(pt_min, pt_max, n_bins + 1)
    
    spine_points = []  # (pseudotime, coord, iloc, idx, class)
    
    for i in range(n_bins):
        bin_mask = (valid_pts >= bin_edges[i]) & (valid_pts < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            bin_mask = (valid_pts >= bin_edges[i]) & (valid_pts <= bin_edges[i + 1])
        
        if not bin_mask.any():
            continue
        
        bin_coords = valid_coords[bin_mask]
        bin_ilocs = valid_ilocs[bin_mask]
        bin_indices = valid_indices[bin_mask]
        bin_pts = valid_pts[bin_mask]
        bin_classes = valid_classes[bin_mask]
        
        # Find medoid (point closest to centroid)
        centroid = bin_coords.mean(axis=0)
        dists_to_centroid = np.linalg.norm(bin_coords - centroid, axis=1)
        medoid_idx = np.argmin(dists_to_centroid)
        
        spine_points.append({
            'pt': bin_pts[medoid_idx],
            'coord': bin_coords[medoid_idx],
            'iloc': bin_ilocs[medoid_idx],
            'idx': bin_indices[medoid_idx],
            'class': bin_classes[medoid_idx],
        })
    
    # Step 2: Greedily select n_total points that:
    #   - Cover all classes
    #   - Are evenly spaced along the spine
    #   - Minimize total path length
    
    # First, ensure each class is represented
    classes_needed = set(classes)
    selected = []
    
    # For each class, find the spine point of that class closest to class centroid
    for cls in classes:
        cls_spine = [sp for sp in spine_points if sp['class'] == cls]
        if not cls_spine:
            continue
        
        # Find the one closest to the overall trajectory
        # (middle of its pseudotime range for that class)
        cls_pts = [sp['pt'] for sp in cls_spine]
        median_pt = np.median(cls_pts)
        best = min(cls_spine, key=lambda sp: abs(sp['pt'] - median_pt))
        
        if best['idx'] not in [s['idx'] for s in selected]:
            selected.append(best)
            classes_needed.discard(cls)
    
    # Sort by pseudotime
    selected.sort(key=lambda x: x['pt'])
    
    # Step 3: Add more points to fill gaps and smooth trajectory
    used_idx = {s['idx'] for s in selected}
    available_spine = [sp for sp in spine_points if sp['idx'] not in used_idx]
    
    while len(selected) < n_total and available_spine:
        selected.sort(key=lambda x: x['pt'])
        
        # Find the segment with largest "detour" (most benefit from adding a point)
        best_improvement = -np.inf
        best_point = None
        
        for gap_idx in range(len(selected) - 1):
            left = selected[gap_idx]
            right = selected[gap_idx + 1]
            
            direct_dist = np.linalg.norm(right['coord'] - left['coord'])
            
            # Find spine points between these two
            candidates = [sp for sp in available_spine 
                         if left['pt'] < sp['pt'] < right['pt']]
            
            if not candidates:
                continue
            
            for cand in candidates:
                via_dist = (np.linalg.norm(cand['coord'] - left['coord']) +
                           np.linalg.norm(right['coord'] - cand['coord']))
                
                # Improvement: how much does adding this point help?
                # Negative value = point is "on the way" (good)
                # We want points where via_dist is close to direct_dist
                detour = via_dist - direct_dist
                
                # Prioritize larger gaps and smaller detours
                gap_size = right['pt'] - left['pt']
                improvement = gap_size - detour * 2
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_point = cand
        
        if best_point:
            selected.append(best_point)
            used_idx.add(best_point['idx'])
            available_spine = [sp for sp in available_spine if sp['idx'] != best_point['idx']]
        else:
            break
    
    # Convert to output format
    selected.sort(key=lambda x: x['pt'])
    results = []
    for sp in selected:
        row = adata.obs.loc[sp['idx']]
        results.append({
            'slide_id': row['slide_id'],
            'patch_id': row['patch_id'],
            'class': row['class'],
            'dpt_pseudotime': row['dpt_pseudotime'],
        })
    
    return results
# def select_patches_by_class(adata, prog_config, n_total=5):
#     """
#     Select patches with:
#     1. At least one patch per class (median pseudotime for that class)
#     2. Additional patches fill gaps in pseudotime coverage
#     3. Preference for patches that form a straighter trajectory in diffusion space
    
#     Returns patches sorted by pseudotime for trajectory visualization.
#     """
#     classes = prog_config["classes"]
#     pseudotime = adata.obs['dpt_pseudotime']
#     valid_mask = np.isfinite(pseudotime)
#     coords = adata.obsm['X_diffmap'][:, 1:3]  # DC1 and DC2
    
#     results = []
#     used_indices = set()
    
#     # Step 1: Guarantee one patch per class (at median pseudotime)
#     for cls in classes:
#         class_mask = (adata.obs['class'] == cls) & valid_mask
#         class_subset = adata.obs[class_mask]
        
#         if class_subset.empty:
#             continue
            
#         median_pt = class_subset['dpt_pseudotime'].median()
#         idx = (class_subset['dpt_pseudotime'] - median_pt).abs().idxmin()
#         row = adata.obs.loc[idx]
        
#         # Get integer location for coords
#         iloc = adata.obs.index.get_loc(idx)
        
#         results.append({
#             'slide_id': row['slide_id'],
#             'patch_id': row['patch_id'],
#             'class': row['class'],
#             'dpt_pseudotime': row['dpt_pseudotime'],
#             '_obs_idx': idx,
#             '_iloc': iloc,
#             '_coord': coords[iloc]
#         })
#         used_indices.add(idx)
    
#     # Sort by pseudotime before adding extras
#     results.sort(key=lambda x: x['dpt_pseudotime'])
    
#     # Step 2: Fill remaining slots with linearity preference
#     n_extra = n_total - len(results)
    
#     if n_extra > 0:
#         pt_min = adata.obs.loc[valid_mask, 'dpt_pseudotime'].min()
#         pt_max = adata.obs.loc[valid_mask, 'dpt_pseudotime'].max()
        
#         for _ in range(n_extra):
#             # Recalculate gaps with current selections
#             results.sort(key=lambda x: x['dpt_pseudotime'])
#             current_pts = [r['dpt_pseudotime'] for r in results]
#             boundaries = [pt_min] + current_pts + [pt_max]
            
#             # Find largest gap
#             gaps = []
#             for i in range(len(boundaries) - 1):
#                 gap_size = boundaries[i + 1] - boundaries[i]
#                 gap_center = (boundaries[i] + boundaries[i + 1]) / 2
#                 gaps.append((gap_size, gap_center, i))  # i = index of left boundary
            
#             gaps.sort(key=lambda x: -x[0])
            
#             for gap_size, target_pt, gap_idx in gaps:
#                 available_mask = valid_mask & (~adata.obs.index.isin(used_indices))
#                 available = adata.obs[available_mask]
                
#                 if available.empty:
#                     break
                
#                 # Find candidates near target pseudotime
#                 pt_tolerance = gap_size * 0.4  # Look within 40% of gap
#                 pt_mask = (available['dpt_pseudotime'] - target_pt).abs() < pt_tolerance
#                 candidates = available[pt_mask] if pt_mask.any() else available
                
#                 # Score candidates by linearity
#                 # Get neighboring points in trajectory
#                 if gap_idx == 0:
#                     # Gap is before first point
#                     neighbor_coord = results[0]['_coord']
#                     score_fn = lambda c: -np.linalg.norm(c - neighbor_coord)  # Just pick closest
#                 elif gap_idx >= len(results):
#                     # Gap is after last point
#                     neighbor_coord = results[-1]['_coord']
#                     score_fn = lambda c: -np.linalg.norm(c - neighbor_coord)
#                 else:
#                     # Gap is between two points - score by collinearity
#                     left_coord = results[gap_idx - 1]['_coord']
#                     right_coord = results[gap_idx]['_coord']
                    
#                     def linearity_score(candidate_coord):
#                         # Distance from candidate to the line between left and right
#                         line_vec = right_coord - left_coord
#                         line_len = np.linalg.norm(line_vec)
#                         if line_len < 1e-8:
#                             return -np.linalg.norm(candidate_coord - left_coord)
#                         line_unit = line_vec / line_len
#                         point_vec = candidate_coord - left_coord
#                         # Project onto line
#                         proj_len = np.dot(point_vec, line_unit)
#                         proj = left_coord + proj_len * line_unit
#                         # Distance from point to its projection (perpendicular distance)
#                         perp_dist = np.linalg.norm(candidate_coord - proj)
#                         # Also penalize if projection is outside segment
#                         outside_penalty = 0
#                         if proj_len < 0:
#                             outside_penalty = abs(proj_len)
#                         elif proj_len > line_len:
#                             outside_penalty = proj_len - line_len
#                         return -(perp_dist + outside_penalty * 0.5)
                    
#                     score_fn = linearity_score
                
#                 # Score all candidates
#                 best_score = -np.inf
#                 best_idx = None
                
#                 for idx in candidates.index:
#                     iloc = adata.obs.index.get_loc(idx)
#                     coord = coords[iloc]
                    
#                     # Combined score: linearity + pseudotime proximity
#                     linearity = score_fn(coord)
#                     pt_proximity = -abs(candidates.loc[idx, 'dpt_pseudotime'] - target_pt)
                    
#                     # Weight linearity more heavily
#                     score = linearity * 2 + pt_proximity
                    
#                     if score > best_score:
#                         best_score = score
#                         best_idx = idx
                
#                 if best_idx is not None:
#                     row = adata.obs.loc[best_idx]
#                     iloc = adata.obs.index.get_loc(best_idx)
                    
#                     results.append({
#                         'slide_id': row['slide_id'],
#                         'patch_id': row['patch_id'],
#                         'class': row['class'],
#                         'dpt_pseudotime': row['dpt_pseudotime'],
#                         '_obs_idx': best_idx,
#                         '_iloc': iloc,
#                         '_coord': coords[iloc]
#                     })
#                     used_indices.add(best_idx)
#                     break
    
#     # Sort by pseudotime for trajectory visualization
#     results.sort(key=lambda x: x['dpt_pseudotime'])
    
#     # Remove internal tracking keys
#     for r in results:
#         r.pop('_obs_idx', None)
#         r.pop('_iloc', None)
#         r.pop('_coord', None)
    
#     return results


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
        try:
            with fs.open(gcs_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            results.append((img, item['slide_id'], item['patch_id'], item['dpt_pseudotime'], item['class']))
        except Exception as e:
            logger.warning(f"Could not load image: {gcs_path}")
            results.append((None, item['slide_id'], item['patch_id'], item['dpt_pseudotime'], item['class']))
    return results


def find_loc(adata, slide_id, patch_id):
    obs = adata.obs
    mask = (obs['slide_id'].astype(str) == str(slide_id)) & \
           (obs['patch_id'].astype(str) == str(patch_id))
    hits = np.where(mask)[0]
    return hits[0] if len(hits) > 0 else None


# --- Drawing Helpers ---
def create_process_box(fig, center_x, center_y, width, height, label, sublabel=None, icon_type='model'):
    """Create a processing step box."""
    box = mpatches.FancyBboxPatch(
        (center_x - width/2, center_y - height/2), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        ec=COLORS['accent'], fc='white', lw=2,
        transform=fig.transFigure, zorder=10
    )
    fig.add_artist(box)
    
    if icon_type == 'model':
        # Neural network layers
        layer_width = width * 0.5
        layer_height = height * 0.07
        layer_gap = height * 0.04
        start_y = center_y + height * 0.15
        for i in range(4):
            layer = mpatches.FancyBboxPatch(
                (center_x - layer_width/2, start_y - i * (layer_height + layer_gap)),
                layer_width, layer_height,
                boxstyle="round,pad=0.001,rounding_size=0.004",
                ec='none', fc=COLORS['accent'], alpha=0.25 + i * 0.15,
                transform=fig.transFigure, zorder=11
            )
            fig.add_artist(layer)
    
    # Single combined label below box
    full_label = f"{label} {sublabel}" if sublabel else label
    fig.text(center_x, center_y - height/2 - 0.018, full_label,
             ha='center', va='top', fontsize=pcfg.FONT_SIZES['annotation'] + 1, fontweight='bold',
             color=COLORS['text_dark'], transform=fig.transFigure)


def draw_arrow(fig, start, end, **kwargs):
    """Draw an arrow between two points."""
    defaults = dict(arrowstyle='-|>,head_length=8,head_width=5', color=COLORS['arrow'], lw=2.2)
    defaults.update(kwargs)
    fig.add_artist(mpatches.FancyArrowPatch(start, end, transform=fig.transFigure, **defaults))


def plot_manifold(ax, adata, prog_config, trajectory_locs, color_by='class', show_trajectory=True):
    """
    Plot diffusion manifold.
    
    Args:
        color_by: 'class' or 'pseudotime'
    """
    coords = adata.obsm['X_diffmap'][:, 1:3]
    classes = prog_config["classes"]
    pseudotime = adata.obs['dpt_pseudotime'].values
    valid_mask = np.isfinite(pseudotime)
    
    if color_by == 'class':
        # Plot by class
        for i, cls in enumerate(classes):
            class_mask = (adata.obs['class'] == cls).values & valid_mask
            if not np.any(class_mask):
                continue
            x, y = coords[class_mask, 0], coords[class_mask, 1]
            color = STAGE_COLORS[i % len(STAGE_COLORS)]
            ax.scatter(x, y, c=[color], s=12, alpha=0.4, edgecolors='none', rasterized=True)
        ax.set_title("Colored by Stage", fontsize=pcfg.FONT_SIZES['axis_label'] + 1, 
                    fontweight='bold', color=COLORS['text_dark'], pad=8)
    else:
        # Plot by pseudotime
        pt_valid = pseudotime[valid_mask]
        pt_norm = (pt_valid - pt_valid.min()) / (pt_valid.max() - pt_valid.min())
        ax.scatter(coords[valid_mask, 0], coords[valid_mask, 1], 
                  c=pt_norm, cmap=PSEUDOTIME_CMAP, s=12, alpha=0.5, edgecolors='none', rasterized=True)
        ax.set_title("Colored by Pseudotime", fontsize=pcfg.FONT_SIZES['axis_label'] + 1, 
                    fontweight='bold', color=COLORS['text_dark'], pad=8)
    
    # Draw trajectory
    if show_trajectory and trajectory_locs:
        traj_data = []
        for loc in trajectory_locs:
            if loc is not None:
                pt = pseudotime[loc]
                if np.isfinite(pt):
                    cls = adata.obs.iloc[loc]['class']
                    class_idx = classes.index(cls) if cls in classes else 0
                    traj_data.append({'x': coords[loc, 0], 'y': coords[loc, 1], 
                                     'pt': pt, 'class_idx': class_idx})
        
        traj_data.sort(key=lambda d: d['pt'])
        
        if len(traj_data) > 1:
            traj_coords = np.array([[d['x'], d['y']] for d in traj_data])
            segments = np.array([[traj_coords[i], traj_coords[i+1]] for i in range(len(traj_coords)-1)])
            n_seg = len(segments)
            seg_colors = [PSEUDOTIME_CMAP(i / max(n_seg - 1, 1)) for i in range(n_seg)]
            lc = LineCollection(segments, colors=seg_colors, linewidth=2.5, alpha=0.9, zorder=10)
            ax.add_collection(lc)
        
        for d in traj_data:
            color = STAGE_COLORS[d['class_idx'] % len(STAGE_COLORS)]
            ax.scatter(d['x'], d['y'], c='white', s=140, edgecolors='none', zorder=14)
            ax.scatter(d['x'], d['y'], c=[color], s=100, edgecolors='white', linewidth=2, zorder=15)
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.8)
    ax.set_facecolor('white')


def plot_violin(ax, adata, prog_config):
    """Plot violin of pseudotime by stage."""
    classes = prog_config["classes"]
    pseudotime = adata.obs['dpt_pseudotime'].values
    valid_mask = np.isfinite(pseudotime)
    
    data_by_class = []
    positions = []
    colors = []
    
    for i, cls in enumerate(classes):
        class_mask = (adata.obs['class'] == cls).values & valid_mask
        if np.any(class_mask):
            data_by_class.append(pseudotime[class_mask])
            positions.append(i)
            colors.append(STAGE_COLORS[i % len(STAGE_COLORS)])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parts = ax.violinplot(data_by_class, positions=positions, 
                              showmeans=False, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('white')
        pc.set_alpha(0.7)
        pc.set_linewidth(1)
    
    parts['cmedians'].set_color(COLORS['text_dark'])
    parts['cmedians'].set_linewidth(2)
    
    # Scatter points
    for i, (data, pos) in enumerate(zip(data_by_class, positions)):
        n_show = min(len(data), 80)
        idx = np.random.choice(len(data), n_show, replace=False) if len(data) > n_show else np.arange(len(data))
        jitter = np.random.uniform(-0.18, 0.18, len(idx))
        ax.scatter(pos + jitter, data[idx], c=[colors[i]], s=6, alpha=0.25, edgecolors='none', rasterized=True)
    
    short_labels = [pcfg.get_class_short_name(cls, inline=True) for cls in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(short_labels, fontsize=pcfg.FONT_SIZES['small'], rotation=25, ha='right')
    ax.set_ylabel("Pseudotime", fontsize=pcfg.FONT_SIZES['annotation'])
    ax.set_xlabel("Disease Stage", fontsize=pcfg.FONT_SIZES['annotation'])
    ax.set_title("Pseudotime Distribution by Stage", fontsize=pcfg.FONT_SIZES['axis_label'], 
                fontweight='bold', color=COLORS['text_dark'], pad=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text_light'])


def create_tau_panel(fig, x, y, width, height, tau_score):
    """Create tau metric panel with explanation."""
    # Background
    box = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        ec=COLORS['accent'], fc='#FAFCFF', lw=2,
        transform=fig.transFigure, zorder=10
    )
    fig.add_artist(box)
    
    # Calculate absolute positions within panel
    panel_center_x = x + width / 2
    top = y + height
    
    # Header
    fig.text(panel_center_x, top - height * 0.08, "Trajectory Fidelity Metric",
             ha='center', va='top', fontsize=pcfg.FONT_SIZES['axis_label'] + 1, 
             fontweight='bold', color=COLORS['text_dark'], transform=fig.transFigure)
    
    # Tau value - large and prominent
    fig.text(panel_center_x, top - height * 0.25, f"τ = {tau_score:.2f}",
             ha='center', va='top', fontsize=pcfg.FONT_SIZES['title'] + 6, 
             fontweight='bold', color=COLORS['accent'], transform=fig.transFigure)
    
    # Interpretation text
    interpretation = (
        "Kendall's τ represents the net ordering concordance:\n"
        "the probability the model correctly ranks a disease\n"
        "stage pair minus the probability it reverses them."
    )
    fig.text(panel_center_x, top - height * 0.48, interpretation,
             ha='center', va='top', fontsize=pcfg.FONT_SIZES['small'] + 0.5, 
             color=COLORS['text_light'], transform=fig.transFigure,
             linespacing=1.4, style='italic')
    
    # Scale bar with gradient
    scale_width = width * 0.65
    scale_left = x + (width - scale_width) / 2
    scale_y = y + height * 0.18
    scale_height = height * 0.08
    
    # Gradient bar
    grad_ax = fig.add_axes([scale_left, scale_y, scale_width, scale_height])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    grad_ax.imshow(gradient, aspect='auto', cmap=PSEUDOTIME_CMAP, extent=[0, 1, 0, 1])
    grad_ax.set_xticks([0, 0.5, 1])
    grad_ax.set_xticklabels(['-1', '0', '+1'], fontsize=pcfg.FONT_SIZES['annotation'], color=COLORS['text_dark'])
    grad_ax.set_yticks([])
    grad_ax.tick_params(length=3, pad=3)
    for spine in grad_ax.spines.values():
        spine.set_visible(False)
    
    # Tau marker
    tau_norm = (tau_score + 1) / 2
    grad_ax.scatter([tau_norm], [0.5], s=250, c='white', edgecolors=COLORS['accent'], 
                   linewidth=3, zorder=15, clip_on=False)
    grad_ax.scatter([tau_norm], [0.5], s=100, c=[COLORS['accent']], zorder=16, clip_on=False)
    
    # Scale endpoint labels
    fig.text(scale_left, scale_y - 0.015, "Reversed", ha='left', va='top',
             fontsize=pcfg.FONT_SIZES['small'], color=COLORS['text_light'], transform=fig.transFigure)
    fig.text(scale_left + scale_width, scale_y - 0.015, "Concordant", ha='right', va='top',
             fontsize=pcfg.FONT_SIZES['small'], color=COLORS['text_light'], transform=fig.transFigure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ICML Figure 1 - Method Overview",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    available_progressions = [p["name"] for p in config.PROGRESSIONS]
    
    parser.add_argument("--progression", "-p", type=str, required=True, choices=available_progressions)
    parser.add_argument("--model", "-m", type=str, default="uni2", choices=config.EXPECTED_MODELS)
    parser.add_argument("--n-patches", "-n", type=int, default=N_PATCHES_DEFAULT)
    parser.add_argument("--output-dir", "-o", type=Path, default=config.PLOTS_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force-one-per-class", action=argparse.BooleanOptionalAction, default=True,
                    help="Selects exactly one representative patch per class instead of sampling by pseudotime.")
    return parser.parse_args()


def main():
    args = parse_args()

    
    prog_config = next((p for p in config.PROGRESSIONS if p["name"] == args.progression), None)
    if prog_config is None:
        logger.error(f"Progression '{args.progression}' not found.")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data for {args.model} on {args.progression}...")
    temp_reg = RegistryConfig(
        bucket=prog_config["bucket"], prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"], models=[args.model], 
        progression_name=args.progression, scan_all_models=False
    )
    temp_ds = ProgressionEmbeddingDataset(temp_reg)
    patch_ids = temp_ds.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=config.EVALUATION["seed"]
    )
    
    adata, _ = get_model_data(args.progression, args.model, patch_ids, prog_config)
    if args.force_one_per_class:
        logger.info("Sampling 1 patch per class (median pseudotime)...")
        selected_patches_info = select_patches_by_class(adata, prog_config)
    else:
        logger.info(f"Sampling {args.n_patches} patches evenly across pseudotime...")
        selected_patches_info = select_patches_by_dpt(adata, n_samples=args.n_patches)   
    image_data = fetch_images_from_selection(selected_patches_info, prog_config)
    
    locs = [find_loc(adata, x[1], x[2]) for x in image_data]
    valid_indices = [i for i, l in enumerate(locs) if l is not None]
    image_data = [image_data[i] for i in valid_indices]
    locs = [locs[i] for i in valid_indices]
    
    tau_score = adata.uns['tau']
    classes = prog_config["classes"]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # === Create Figure - Single Row Layout ===
    fig = plt.figure(figsize=(18, 5.2), facecolor='white')
    
    # Main content area (expanded since no title)
    content_top = 0.94
    content_bottom = 0.06
    content_center = (content_top + content_bottom) / 2
    
    # --- Input Patches (left section) ---
    patch_left = 0.02
    patch_width = 0.055
    patch_height = 0.22
    patch_gap_x = 0.012  # Increased horizontal gap
    patch_gap_y = 0.06   # Increased vertical gap for labels
    
    n_patches = len(image_data)
    n_top = (n_patches + 1) // 2
    n_bottom = n_patches - n_top
    
    # Calculate positions - ensure no overlap
    patch_area_height = 2 * patch_height + patch_gap_y
    patch_top_y = content_center + patch_gap_y/2 + 0.02
    patch_bottom_y = patch_top_y - patch_height - patch_gap_y
    # Center bottom row under top row
    bottom_offset = (n_top - n_bottom) * (patch_width + patch_gap_x) / 2
    
    # Section label - aligned with left edge of patches
    fig.text(patch_left, patch_top_y + patch_height + 0.025, "Input Patches",
             fontsize=pcfg.FONT_SIZES['section_header'] + 2, fontweight='bold', color=COLORS['text_dark'])
    
    for i, (img, slide_id, patch_id, pt, cls) in enumerate(image_data):
        if i % 2 == 0:
            row_idx = i // 2
            ax_x = patch_left + row_idx * (patch_width + patch_gap_x)
            ax_y = patch_top_y
        else:
            row_idx = i // 2
            ax_x = patch_left + bottom_offset + row_idx * (patch_width + patch_gap_x)
            ax_y = patch_bottom_y
        
        ax = fig.add_axes([ax_x, ax_y, patch_width, patch_height])
        if img:
            ax.imshow(img)
        
        class_idx = class_to_idx.get(cls, 0)
        border_color = STAGE_COLORS[class_idx % len(STAGE_COLORS)]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Use short abbreviated labels under patches
        short_label = pcfg.get_class_short_name(cls, inline=True)
        # Further abbreviate if needed
        if short_label == "Actinic Keratosis":
            short_label = "AK"
        elif short_label == "Ca in Situ":
            short_label = "CIS"
        fig.text(ax_x + patch_width/2, ax_y - 0.018, short_label,
                ha='center', va='top', fontsize=pcfg.FONT_SIZES['small'] + 1,
                color=COLORS['text_dark'], transform=fig.transFigure)
    
    # Calculate where patches end
    patches_right_edge = patch_left + n_top * (patch_width + patch_gap_x)
    
    # --- Foundation Model Box (between patches and container) ---
    model_box_width = 0.05
    model_box_height = 0.28
    model_x = patches_right_edge + 0.04 + model_box_width/2  # Center of model box
    model_y = content_center + 0.02
    
    # Arrow from patches to model (symmetric gap on both sides)
    arrow_gap = 0.012
    draw_arrow(fig, (patches_right_edge + arrow_gap, model_y), (model_x - model_box_width/2 - arrow_gap, model_y))
    create_process_box(fig, model_x, model_y, model_box_width, model_box_height, "Foundation", sublabel="Model", icon_type='model')
    
    # === Diffusion Pseudotime Analysis Container ===
    container_left = model_x + model_box_width/2 + 0.04
    container_right = 0.98
    container_width = container_right - container_left
    container_bottom = content_bottom
    container_top = content_top
    container_height = container_top - container_bottom
    
    # Arrow from model to container (symmetric gap)
    draw_arrow(fig, (model_x + model_box_width/2 + arrow_gap, model_y), (container_left - arrow_gap, model_y))
    
    # Container box
    container_box = mpatches.FancyBboxPatch(
        (container_left, container_bottom),
        container_width, container_height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        ec=COLORS['accent'], fc='#F8FBFF', lw=2,
        transform=fig.transFigure, zorder=0
    )
    fig.add_artist(container_box)
    
    # Container title
    fig.text(container_left + container_width/2, container_top - 0.015,
             "Diffusion Pseudotime Analysis",
             ha='center', va='top', fontsize=pcfg.FONT_SIZES['section_header'] + 2, 
             fontweight='bold', color=COLORS['accent'])
    
    # --- Two Manifold Plots (left portion of container) ---
    manifold_width = 0.185
    manifold_height = 0.52
    manifold_gap = 0.025
    manifold1_left = container_left + 0.02
    manifold2_left = manifold1_left + manifold_width + manifold_gap
    manifold_bottom = container_bottom + 0.12
    
    ax_class = fig.add_axes([manifold1_left, manifold_bottom, manifold_width, manifold_height])
    plot_manifold(ax_class, adata, prog_config, locs, color_by='class')
    ax_class.set_xlabel("DC 1", fontsize=pcfg.FONT_SIZES['annotation'] + 1, color=COLORS['text_light'], labelpad=2)
    ax_class.set_ylabel("DC 2", fontsize=pcfg.FONT_SIZES['annotation'] + 1, color=COLORS['text_light'], labelpad=2)
    
    ax_pt = fig.add_axes([manifold2_left, manifold_bottom, manifold_width, manifold_height])
    plot_manifold(ax_pt, adata, prog_config, locs, color_by='pseudotime')
    ax_pt.set_xlabel("DC 1", fontsize=pcfg.FONT_SIZES['annotation'] + 1, color=COLORS['text_light'], labelpad=2)
    ax_pt.set_ylabel("DC 2", fontsize=pcfg.FONT_SIZES['annotation'] + 1, color=COLORS['text_light'], labelpad=2)
    
    # --- Legend Row (below manifolds) - Two rows for better spacing ---
    legend_y1 = container_bottom + 0.06
    legend_y2 = container_bottom + 0.025
    
    # Helper to get abbreviated class name
    def get_abbrev(cls_name):
        short = pcfg.get_class_short_name(cls_name, inline=True)
        if short == "Actinic Keratosis":
            return "AK"
        elif short == "Ca in Situ":
            return "CIS"
        return short
    
    # Stage legend - split into two rows
    fig.text(manifold1_left, legend_y1, "Stage:",
             fontsize=pcfg.FONT_SIZES['small'] + 1, fontweight='medium', color=COLORS['text_dark'], va='center')
    
    # First two stages on row 1
    x_offset = manifold1_left + 0.042
    for i in range(min(2, len(classes))):
        cls = classes[i]
        color = STAGE_COLORS[i % len(STAGE_COLORS)]
        marker_ax = fig.add_axes([x_offset, legend_y1 - 0.009, 0.013, 0.018])
        marker_ax.scatter([0.5], [0.5], c=[color], s=55, edgecolors='white', linewidth=0.8)
        marker_ax.set_xlim(0, 1); marker_ax.set_ylim(0, 1); marker_ax.axis('off')
        short_name = get_abbrev(cls)
        fig.text(x_offset + 0.016, legend_y1, short_name,
                fontsize=pcfg.FONT_SIZES['small'], color=COLORS['text_light'], va='center')
        x_offset += 0.05 + len(short_name) * 0.004
    
    # Remaining stages on row 2
    x_offset = manifold1_left + 0.042
    for i in range(2, len(classes)):
        cls = classes[i]
        color = STAGE_COLORS[i % len(STAGE_COLORS)]
        marker_ax = fig.add_axes([x_offset, legend_y2 - 0.009, 0.013, 0.018])
        marker_ax.scatter([0.5], [0.5], c=[color], s=55, edgecolors='white', linewidth=0.8)
        marker_ax.set_xlim(0, 1); marker_ax.set_ylim(0, 1); marker_ax.axis('off')
        short_name = get_abbrev(cls)
        fig.text(x_offset + 0.016, legend_y2, short_name,
                fontsize=pcfg.FONT_SIZES['small'], color=COLORS['text_light'], va='center')
        x_offset += 0.05 + len(short_name) * 0.004
    
    # Pseudotime legend
    # pt_x = manifold2_left + 0.01
    # fig.text(pt_x, legend_y1, "Pseudotime:",
    #          fontsize=pcfg.FONT_SIZES['small'] + 1, fontweight='medium', color=COLORS['text_dark'], va='center')
    # grad_width = 0.08
    # grad_ax = fig.add_axes([pt_x + 0.058, legend_y1 - 0.008, grad_width, 0.016])
    # gradient = np.linspace(0, 1, 256).reshape(1, -1)
    # grad_ax.imshow(gradient, aspect='auto', cmap=PSEUDOTIME_CMAP)
    # grad_ax.axis('off')
    # fig.text(pt_x + 0.054, legend_y1, "0", fontsize=pcfg.FONT_SIZES['small'], 
    #          ha='right', va='center', color=COLORS['text_light'])
    # fig.text(pt_x + 0.058 + grad_width + 0.005, legend_y1, "1", 
    #          fontsize=pcfg.FONT_SIZES['small'], ha='left', va='center', color=COLORS['text_light'])
    
    # Pseudotime legend
    pt_x = manifold2_left + 0.01
    fig.text(pt_x, legend_y1, "Pseudotime:",
             fontsize=pcfg.FONT_SIZES['small'] + 1, fontweight='medium', color=COLORS['text_dark'], va='center')
    
    grad_width = 0.08
    grad_left = pt_x + 0.09  # Start gradient after "Pseudotime:" label
    
    grad_ax = fig.add_axes([grad_left, legend_y1 - 0.008, grad_width, 0.016])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    grad_ax.imshow(gradient, aspect='auto', cmap=PSEUDOTIME_CMAP)
    grad_ax.axis('off')
    
    # "0" label to the left of gradient
    fig.text(grad_left - 0.006, legend_y1, "0", fontsize=pcfg.FONT_SIZES['small'], 
             ha='right', va='center', color=COLORS['text_light'])
    # "1" label to the right of gradient
    fig.text(grad_left + grad_width + 0.006, legend_y1, "1", 
             fontsize=pcfg.FONT_SIZES['small'], ha='left', va='center', color=COLORS['text_light'])
    # --- Tau Panel (right portion of container) ---
    tau_left = manifold2_left + manifold_width + 0.035
    tau_width = container_right - tau_left - 0.02
    tau_center_x = tau_left + tau_width / 2
    
    # Vertical divider line
    divider_x = tau_left - 0.018
    fig.add_artist(plt.Line2D([divider_x, divider_x], 
                              [container_bottom + 0.05, container_top - 0.05], 
                              color=COLORS['grid'], lw=1.5, transform=fig.transFigure))
    
    # Large tau value (main focal point)
    fig.text(tau_center_x, container_top - 0.12, f"τ = {tau_score:.2f}",
             ha='center', va='top', fontsize=pcfg.FONT_SIZES['title'] + 14, 
             fontweight='bold', color=COLORS['accent'])
    
    # Subtitle
    fig.text(tau_center_x, container_top - 0.30, "Trajectory Fidelity",
             ha='center', va='top', fontsize=pcfg.FONT_SIZES['axis_label'] + 2, 
             fontweight='medium', color=COLORS['text_dark'])
    
    # Interpretation text - clearer explanation
    interpretation = (
    "Fraction of sample pairs\n"
    "where pseudotime agrees with\n"
    "stage order, minus those\n"
    "that disagree, normalized.\n"
    "1 = perfect ordering\n"
    "0 = random\n"
    "-1 = reversed"
)
    fig.text(tau_center_x, container_top - 0.44, interpretation,
             ha='center', va='top', fontsize=pcfg.FONT_SIZES['axis_label'] + 1, 
             color=COLORS['text_light'], linespacing=1.5, style='italic')
    
    # === Save ===
    output_subdir = args.output_dir / "method_overview"
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / f"method_{args.progression}_{args.model}.png"
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ Saved to {output_path}")
    
    plt.close()
    return output_path


if __name__ == "__main__":
    main()