#!/usr/bin/env python
"""
Trajectory Inspector with HistoPLUS Deconstruction.

1. Loads a specific model (e.g., UNI-2) for a specific progression (e.g., SCC).
2. Computes the Diffusion Pseudotime (DPT) manifold.
3. Samples patches along the manifold using quantile sampling.
4. Runs HistoPLUS inference on these patches (upsampling 20x -> 40x).
5. Generates composite figures: Raw Image | Segmentation | Quantitative Trends.

Usage:
    python inspect_trajectory_histoplus.py --model uni2 --progression SCC
    python inspect_trajectory_histoplus.py --model uni2 --progression SCC --seed 42
    python inspect_trajectory_histoplus.py --seed 12345 --jitter 0.1
"""

import sys
import os
import argparse
from pathlib import Path
import logging
import time
import numpy as np
import pandas as pd
import torch

# Plotting & Image Imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import fsspec
from skimage.segmentation import find_boundaries
from skimage.measure import label

# HistoPLUS Imports
try:
    from histoplus.helpers.segmentor import CellViTSegmentor
    HISTOPLUS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: HistoPLUS not found. Install with: pip install histoplus")
    HISTOPLUS_AVAILABLE = False

# Ensure we can import config from parent directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("❌ Critical Error: Could not import 'config.py'.")
    sys.exit(1)

from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import (
    DPTConfig,
    DPTMetric,
    cohort_to_anndata,
    compute_dpt,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "uni2"
DEFAULT_PROGRESSION = "SCC"
DEFAULT_NUM_SAMPLES = 8
DEFAULT_JITTER = 0.05

EMBEDDING_TYPE = "final_embedding"
INFERENCE_SIZE = 448

# HistoPLUS Class Mapping (0 = background, 1-13 = cell types)
HISTOPLUS_CLASSES = {
    1: "Cancer cells",
    2: "Lymphocytes",
    3: "Fibroblasts",
    4: "Plasmocytes",
    5: "Eosinophils",
    6: "Neutrophils",
    7: "Macrophages",
    8: "Smooth muscle",
    9: "Endothelial",
    10: "Red blood cells",
    11: "Epithelial (Normal)",
    12: "Mitotic Figures",
    13: "Apoptotic Bodies"
}

# Color Palette for visualization (RGB tuples, 0-1 range)
CLASS_COLORS = {
    "Cancer cells": (0.9, 0.1, 0.1),        # Red
    "Epithelial (Normal)": (0.0, 0.8, 0.8), # Cyan
    "Fibroblasts": (0.2, 0.8, 0.2),         # Green
    "Lymphocytes": (0.1, 0.1, 0.9),         # Blue
    "Macrophages": (0.6, 0.1, 0.6),         # Purple
    "Neutrophils": (1.0, 0.5, 0.0),         # Orange
    "Eosinophils": (1.0, 0.2, 0.8),         # Magenta
    "Plasmocytes": (0.4, 0.4, 0.8),         # Light purple
    "Smooth muscle": (0.6, 0.4, 0.2),       # Brown
    "Endothelial": (0.2, 0.6, 0.4),         # Teal
    "Red blood cells": (0.8, 0.2, 0.2),     # Dark red
    "Mitotic Figures": (1.0, 1.0, 0.0),     # Yellow
    "Apoptotic Bodies": (0.3, 0.3, 0.3),    # Dark grey
    "Background": (0.9, 0.9, 0.9),          # Light grey
}

# Key classes for the summary figure
KEY_CLASSES_FOR_PLOT = [
    "Cancer cells", 
    "Epithelial (Normal)", 
    "Fibroblasts", 
    "Lymphocytes",
]

# All classes for the detailed figure (excluding background)
ALL_CLASSES_FOR_PLOT = list(HISTOPLUS_CLASSES.values())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_hf_token():
    """Load HuggingFace token from environment or file."""
    token = os.getenv("HF_TOKEN")
    if not token:
        token_path = Path("~/.hf_token").expanduser()
        if token_path.exists():
            token = token_path.read_text().strip()
    
    if token:
        os.environ["HF_TOKEN"] = token
        logger.info("✓ HuggingFace token loaded")
    else:
        logger.warning("⚠️ No HF_TOKEN found. HistoPLUS download may fail.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect DPT trajectory with HistoPLUS cellular analysis."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL,
        choices=config.EXPECTED_MODELS,
        help=f"Foundation model to analyze (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--progression", 
        type=str, 
        default=DEFAULT_PROGRESSION,
        help=f"Disease progression (default: {DEFAULT_PROGRESSION})"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=DEFAULT_NUM_SAMPLES, 
        help=f"Patches to sample along trajectory (default: {DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None = different each run)"
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=DEFAULT_JITTER,
        help=f"Randomness range around each quantile point (default: {DEFAULT_JITTER})"
    )
    return parser.parse_args()


# =============================================================================
# SAMPLING FUNCTIONS
# =============================================================================

def sample_patches_quantile(
    trajectory_df: pd.DataFrame, 
    num_samples: int,
    rng: np.random.Generator, 
    jitter: float = 0.05
) -> pd.DataFrame:
    """
    Sample patches at evenly-spaced RANKS (percentiles) with optional jitter.
    
    This adapts to the actual distribution of data - if most patches cluster
    in early pseudotime, we'll still sample across all stages proportionally.
    
    Args:
        trajectory_df: DataFrame sorted by dpt_pseudotime_norm
        num_samples: Number of patches to select
        rng: Numpy random generator
        jitter: Random range around each target point (0 = exact, 0.1 = ±10% of data)
        
    Returns:
        Selected patches DataFrame
    """
    n_total = len(trajectory_df)
    target_ranks = np.linspace(0, n_total - 1, num_samples).astype(int)
    
    selected_ilocs = []
    for rank in target_ranks:
        # Add jitter in rank space
        if jitter > 0:
            max_offset = int(n_total * jitter)
            if max_offset > 0:
                offset = rng.integers(-max_offset, max_offset + 1)
                rank_jittered = np.clip(rank + offset, 0, n_total - 1)
            else:
                rank_jittered = rank
        else:
            rank_jittered = rank
        
        selected_ilocs.append(rank_jittered)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ilocs = [x for x in selected_ilocs if not (x in seen or seen.add(x))]
    
    return trajectory_df.iloc[unique_ilocs].copy()


# =============================================================================
# HISTOPLUS INFERENCE
# =============================================================================

class HistoPLUSInference:
    """
    Wrapper for HistoPLUS model inference on individual patches.
    """
    
    def __init__(self):
        if not HISTOPLUS_AVAILABLE:
            raise RuntimeError("HistoPLUS not installed")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing HistoPLUS on {self.device}...")
        
        self.segmentor = CellViTSegmentor.from_histoplus(
            mpp=0.25,
            mixed_precision=(self.device == "cuda"),
            inference_image_size=INFERENCE_SIZE,
        )
        
        if hasattr(self.segmentor, 'model'):
            self.segmentor.model = self.segmentor.model.to(self.device)
            self.segmentor.model.eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        logger.info("✓ HistoPLUS ready")

    def preprocess(self, img_pil: Image.Image) -> tuple:
        """Preprocess PIL image for model input."""
        img_resized = img_pil.resize(
            (INFERENCE_SIZE, INFERENCE_SIZE), 
            Image.Resampling.LANCZOS
        )
        
        img_arr = np.array(img_resized)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        
        return img_tensor, img_resized

    @torch.no_grad()
    def predict(self, img_pil: Image.Image) -> tuple:
        """Run inference on a single patch."""
        img_tensor, img_resized = self.preprocess(img_pil)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        outputs = self.segmentor.model(img_tensor)
        
        if "tp" not in outputs:
            raise KeyError(f"Expected key 'tp', found: {list(outputs.keys())}")
        
        type_logits = outputs["tp"]
        type_map = torch.argmax(type_logits, dim=1).cpu().numpy()[0]
        
        return img_resized, type_map


def get_class_pixel_fractions(type_map: np.ndarray) -> dict:
    """Calculate the fraction of pixels belonging to each cell class."""
    total_pixels = type_map.size
    fractions = {}
    
    for class_idx, class_name in HISTOPLUS_CLASSES.items():
        mask = (type_map == class_idx)
        pixel_count = mask.sum()
        fractions[class_name] = (pixel_count / total_pixels) * 100
    
    bg_mask = (type_map == 0)
    fractions["Background"] = (bg_mask.sum() / total_pixels) * 100
    
    return fractions


def create_overlay(img_pil: Image.Image, type_map: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Create a visualization overlay showing cell type predictions."""
    img_arr = np.array(img_pil).astype(np.float32)
    label_rgb = np.zeros_like(img_arr)
    
    for class_idx, class_name in HISTOPLUS_CLASSES.items():
        mask = (type_map == class_idx)
        if not np.any(mask):
            continue
        
        color = CLASS_COLORS.get(class_name, (0.5, 0.5, 0.5))
        for c in range(3):
            label_rgb[mask, c] = color[c] * 255
    
    foreground_mask = (type_map > 0)
    overlay = img_arr.copy()
    
    if np.any(foreground_mask):
        overlay[foreground_mask] = (
            (1 - alpha) * img_arr[foreground_mask] + 
            alpha * label_rgb[foreground_mask]
        )
    
    boundaries = find_boundaries(type_map, mode='thin')
    overlay[boundaries] = [255, 255, 255]
    
    return Image.fromarray(overlay.astype(np.uint8))


# =============================================================================
# TRAJECTORY COMPUTATION
# =============================================================================

def compute_trajectory(
    prog_config: dict, 
    model_name: str, 
    num_samples: int,
    rng: np.random.Generator,
    jitter: float
) -> pd.DataFrame:
    """Compute DPT trajectory and select patches using quantile sampling."""
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=[model_name],
        progression_name=prog_config["name"],
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    
    try:
        patch_ids = dataset.sample_patch_ids(
            n_per_class=config.EVALUATION["n_per_class"],
            max_per_slide=config.EVALUATION["max_per_slide"],
            seed=config.EVALUATION["seed"]
        )
        
        dataset.load_model_into_memory(model_name, patch_ids)
        cohort_df = dataset.get_cohort(patch_ids, model_name, EMBEDDING_TYPE)
        
        if cohort_df.empty:
            raise ValueError(f"No data found for {model_name}")
        
        adata = cohort_to_anndata(cohort_df, prog_config["classes"])
        
        dpt_conf = DPTConfig(
            n_neighbors=config.DPT["n_neighbors"],
            n_diffusion_components=config.DPT["n_diffusion_components"],
            metrics={DPTMetric.TAU},
            subsample_size=2000
        )
        
        logger.info("⚙️  Computing Diffusion Pseudotime...")
        result = compute_dpt(adata, prog_config["root_class"], dpt_conf)
        logger.info(f"   Trajectory Fidelity: τ = {result.tau:.3f}")
        
        if "dpt_pseudotime" not in adata.obs:
            raise ValueError("DPT computation failed")
        
        trajectory_df = adata.obs.copy()
        trajectory_df = trajectory_df[np.isfinite(trajectory_df["dpt_pseudotime"])]
        
        pt_min = trajectory_df["dpt_pseudotime"].min()
        pt_max = trajectory_df["dpt_pseudotime"].max()
        trajectory_df["dpt_pseudotime_norm"] = (
            (trajectory_df["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)
        )
        
        trajectory_df = trajectory_df.sort_values("dpt_pseudotime_norm")
        
        # Use quantile sampling with jitter
        selected_df = sample_patches_quantile(trajectory_df, num_samples, rng, jitter)
        selected_df['tau'] = result.tau
        
        return selected_df
        
    finally:
        dataset.clear_cache()


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_figure_key_classes(
    patches_data: list,
    model_name: str,
    progression_name: str,
    tau: float,
    seed: int,
    output_path: Path
):
    """Generate figure with key cell types only."""
    n = len(patches_data)
    
    fig = plt.figure(figsize=(n * 2.5, 9))
    gs = GridSpec(3, n, height_ratios=[1, 1, 0.9], hspace=0.15, wspace=0.08)
    
    pseudotimes = []
    trends = {cls: [] for cls in KEY_CLASSES_FOR_PLOT}
    
    for i, pdata in enumerate(patches_data):
        pt = pdata['pt']
        pseudotimes.append(pt)
        
        # Row 1: Original Image
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(pdata['orig_img'])
        ax_orig.axis('off')
        
        stage_name = pdata['stage']
        if len(stage_name) > 20:
            stage_name = stage_name[:18] + "..."
        ax_orig.set_title(f"PT: {pt:.2f}\n{stage_name}", fontsize=9, fontweight='bold')
        
        # Row 2: Segmentation Overlay
        ax_seg = fig.add_subplot(gs[1, i])
        ax_seg.imshow(pdata['overlay_img'])
        ax_seg.axis('off')
        
        for cls in KEY_CLASSES_FOR_PLOT:
            value = pdata['fractions'].get(cls, 0)
            trends[cls].append(value)
    
    # Row 3: Trend Plot
    ax_trend = fig.add_subplot(gs[2, :])
    
    for cls in KEY_CLASSES_FOR_PLOT:
        color = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))
        y_vals = trends[cls]
        
        if max(y_vals) > 0.1:
            ax_trend.plot(
                pseudotimes, y_vals,
                marker='o', markersize=6,
                linewidth=2, 
                label=cls,
                color=color,
                alpha=0.85
            )
    
    ax_trend.set_xlabel("Diffusion Pseudotime (Normalized)", fontsize=11)
    ax_trend.set_ylabel("% Tissue Area", fontsize=11)
    ax_trend.set_xlim(-0.05, 1.05)
    ax_trend.set_ylim(bottom=0)
    ax_trend.grid(True, linestyle='--', alpha=0.3)
    ax_trend.spines['top'].set_visible(False)
    ax_trend.spines['right'].set_visible(False)
    
    ax_trend.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(KEY_CLASSES_FOR_PLOT), 5),
        frameon=False,
        fontsize=9
    )
    
    seed_str = f"seed={seed}" if seed is not None else "random"
    fig.suptitle(
        f"Cellular Composition Along Trajectory: {progression_name}\n"
        f"Model: {model_name} (τ = {tau:.2f}) | {seed_str}",
        fontsize=12,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Key classes figure saved to {output_path}")


def generate_figure_all_classes(
    patches_data: list,
    model_name: str,
    progression_name: str,
    tau: float,
    seed: int,
    output_path: Path
):
    """Generate figure with ALL cell types plotted."""
    n = len(patches_data)
    
    # Taller figure to accommodate more legend entries
    fig = plt.figure(figsize=(n * 2.5, 11))
    gs = GridSpec(3, n, height_ratios=[1, 1, 1.2], hspace=0.15, wspace=0.08)
    
    pseudotimes = []
    trends = {cls: [] for cls in ALL_CLASSES_FOR_PLOT}
    
    for i, pdata in enumerate(patches_data):
        pt = pdata['pt']
        pseudotimes.append(pt)
        
        # Row 1: Original Image
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(pdata['orig_img'])
        ax_orig.axis('off')
        
        stage_name = pdata['stage']
        if len(stage_name) > 20:
            stage_name = stage_name[:18] + "..."
        ax_orig.set_title(f"PT: {pt:.2f}\n{stage_name}", fontsize=9, fontweight='bold')
        
        # Row 2: Segmentation Overlay
        ax_seg = fig.add_subplot(gs[1, i])
        ax_seg.imshow(pdata['overlay_img'])
        ax_seg.axis('off')
        
        for cls in ALL_CLASSES_FOR_PLOT:
            value = pdata['fractions'].get(cls, 0)
            trends[cls].append(value)
    
    # Row 3: Trend Plot with ALL classes
    ax_trend = fig.add_subplot(gs[2, :])
    
    # Sort classes by max value for cleaner legend ordering
    class_max_vals = [(cls, max(trends[cls])) for cls in ALL_CLASSES_FOR_PLOT]
    sorted_classes = [cls for cls, _ in sorted(class_max_vals, key=lambda x: -x[1])]
    
    for cls in sorted_classes:
        color = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))
        y_vals = trends[cls]
        
        # Only plot if there's meaningful signal (>0.5%)
        if max(y_vals) > 0.5:
            ax_trend.plot(
                pseudotimes, y_vals,
                marker='o', markersize=5,
                linewidth=1.5, 
                label=f"{cls} (max: {max(y_vals):.1f}%)",
                color=color,
                alpha=0.8
            )
    
    ax_trend.set_xlabel("Diffusion Pseudotime (Normalized)", fontsize=11)
    ax_trend.set_ylabel("% Tissue Area", fontsize=11)
    ax_trend.set_xlim(-0.05, 1.05)
    ax_trend.set_ylim(bottom=0)
    ax_trend.grid(True, linestyle='--', alpha=0.3)
    ax_trend.spines['top'].set_visible(False)
    ax_trend.spines['right'].set_visible(False)
    
    # Legend with more columns and smaller font for all classes
    ax_trend.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=True,
        fontsize=8,
        fancybox=True
    )
    
    seed_str = f"seed={seed}" if seed is not None else "random"
    fig.suptitle(
        f"All Cell Types Along Trajectory: {progression_name}\n"
        f"Model: {model_name} (τ = {tau:.2f}) | {seed_str}",
        fontsize=12,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ All classes figure saved to {output_path}")


def print_summary_table(patches_data: list):
    """Print a summary table of cell compositions."""
    print("\n" + "=" * 120)
    print("CELLULAR COMPOSITION SUMMARY")
    print("=" * 120)
    
    # Header with key classes
    header = f"{'PT':<8} | {'Stage':<25} | "
    header += " | ".join([f"{cls[:12]:<12}" for cls in KEY_CLASSES_FOR_PLOT])
    print(header)
    print("-" * 120)
    
    for pdata in patches_data:
        row = f"{pdata['pt']:<8.2f} | {pdata['stage'][:25]:<25} | "
        values = [f"{pdata['fractions'].get(cls, 0):<12.1f}" for cls in KEY_CLASSES_FOR_PLOT]
        row += " | ".join(values)
        print(row)
    
    print("=" * 120)
    
    # Also print all classes with any signal
    print("\nALL DETECTED CELL TYPES (max > 0.5%):")
    print("-" * 60)
    
    all_fractions = {cls: [] for cls in ALL_CLASSES_FOR_PLOT}
    for pdata in patches_data:
        for cls in ALL_CLASSES_FOR_PLOT:
            all_fractions[cls].append(pdata['fractions'].get(cls, 0))
    
    for cls in ALL_CLASSES_FOR_PLOT:
        max_val = max(all_fractions[cls])
        if max_val > 0.5:
            print(f"  {cls:<25}: max={max_val:>6.1f}%, mean={np.mean(all_fractions[cls]):>6.1f}%")
    
    print("=" * 120)
    print("Note: Values are % of tissue area. HistoPLUS not trained on skin - interpret with caution.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    load_hf_token()
    args = parse_args()
    
    target_model = args.model
    target_progression = args.progression
    num_samples = args.num_samples
    jitter = args.jitter
    
    # Set up random seed
    if args.seed is not None:
        seed = args.seed
        logger.info(f"🎲 Using fixed seed: {seed}")
    else:
        seed = int(time.time() * 1000) % (2**32)
        logger.info(f"🎲 Using random seed: {seed} (pass --seed {seed} to reproduce)")
    
    rng = np.random.default_rng(seed)
    
    # Load progression config
    prog_config = next(
        (p for p in config.PROGRESSIONS if p["name"] == target_progression),
        None
    )
    
    if not prog_config:
        available = [p["name"] for p in config.PROGRESSIONS]
        logger.error(f"❌ Progression '{target_progression}' not found.")
        logger.error(f"   Available: {available}")
        sys.exit(1)
    
    image_subdir = prog_config.get("image_subdir", "imagenet")
    
    logger.info(f"🔬 Analyzing: {target_model} on {target_progression}")
    logger.info(f"   Sampling {num_samples} patches (quantile, jitter={jitter})")
    
    if not HISTOPLUS_AVAILABLE:
        logger.error("❌ HistoPLUS required. Install from: https://github.com/owkin/histoplus")
        sys.exit(1)
    
    try:
        histoplus = HistoPLUSInference()
    except Exception as e:
        logger.error(f"❌ Failed to initialize HistoPLUS: {e}")
        sys.exit(1)
    
    # Compute trajectory with quantile sampling
    logger.info("\n📊 Computing trajectory...")
    selected_df = compute_trajectory(prog_config, target_model, num_samples, rng, jitter)
    tau = selected_df['tau'].iloc[0]
    
    # Process patches
    logger.info(f"\n🔍 Processing {len(selected_df)} patches with HistoPLUS...")
    
    fs = fsspec.filesystem("gcs")
    patches_data = []
    
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        cls = row['class']
        slide_id = str(row['slide_id'])
        patch_id = str(row['patch_id'])
        
        if not slide_id.startswith("slide_"):
            slide_id = f"slide_{slide_id}"
        if not patch_id.startswith("patch_"):
            patch_id = f"patch_{patch_id}"
        
        gcs_path = f"gs://{prog_config['bucket']}/{image_subdir}/train/{cls}/{slide_id}/{patch_id}.png"
        
        try:
            with fs.open(gcs_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            
            img_resized, type_map = histoplus.predict(img)
            overlay = create_overlay(img_resized, type_map)
            fractions = get_class_pixel_fractions(type_map)
            
            patches_data.append({
                'pt': row['dpt_pseudotime_norm'],
                'orig_img': img_resized,
                'overlay_img': overlay,
                'fractions': fractions,
                'stage': cls
            })
            
            cancer_pct = fractions.get("Cancer cells", 0)
            epithelial_pct = fractions.get("Epithelial (Normal)", 0)
            logger.info(
                f"   [{idx+1}/{len(selected_df)}] PT={row['dpt_pseudotime_norm']:.2f} | "
                f"Cancer: {cancer_pct:.1f}% | Epithelial: {epithelial_pct:.1f}%"
            )
            
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to process {patch_id}: {e}")
            continue
    
    if not patches_data:
        logger.error("❌ No patches processed successfully.")
        sys.exit(1)
    
    # Print summary table
    print_summary_table(patches_data)
    
    # Generate BOTH figures
    plot_dir = config.RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    seed_suffix = f"_seed{seed}"
    base_filename = f"histoplus_{target_model}_{target_progression.replace(' ', '_')}{seed_suffix}"
    
    # Figure 1: Key classes only
    logger.info(f"🎨 Generating key classes figure...")
    generate_figure_key_classes(
        patches_data,
        target_model,
        target_progression,
        tau,
        seed,
        plot_dir / f"{base_filename}_key.png"
    )
    
    # Figure 2: All classes
    logger.info(f"🎨 Generating all classes figure...")
    generate_figure_all_classes(
        patches_data,
        target_model,
        target_progression,
        tau,
        seed,
        plot_dir / f"{base_filename}_all.png"
    )
    
    # Summary
    print(f"\n✅ Analysis complete!")
    print(f"   Model: {target_model} (τ = {tau:.2f})")
    print(f"   Progression: {target_progression}")
    print(f"   Seed: {seed}")
    print(f"\n   Outputs:")
    print(f"     Key classes: {plot_dir / f'{base_filename}_key.png'}")
    print(f"     All classes: {plot_dir / f'{base_filename}_all.png'}")
    print(f"\n💡 To reproduce: --seed {seed}")


if __name__ == "__main__":
    main()