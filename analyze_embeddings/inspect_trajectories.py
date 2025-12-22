#!/usr/bin/env python
"""
Trajectory Inspector with HistoPLUS Deconstruction.

1. Loads a specific model (e.g., UNI-2) for a specific progression (e.g., SCC).
2. Computes the Diffusion Pseudotime (DPT) manifold.
3. Samples patches evenly along the manifold (Early -> Late).
4. Runs HistoPLUS inference on these patches (upsampling 20x -> 40x).
5. Generates a composite figure: Raw Image | Segmentation | Quantitative Trends.

Note: This is EXPLORATORY analysis. HistoPLUS was not trained on skin tissue,
and we use semantic segmentation (pixel fractions) rather than instance counts
due to the complexity of proper post-processing.

Usage:
    python inspect_trajectory_histoplus.py --model uni2 --progression SCC
"""

import sys
import os
import argparse
from pathlib import Path
import logging
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

# Default analysis parameters
DEFAULT_MODEL = "uni2"
DEFAULT_PROGRESSION = "SCC"
DEFAULT_NUM_SAMPLES = 8

EMBEDDING_TYPE = "final_embedding"
INFERENCE_SIZE = 448  # HistoPLUS expects ~40x resolution

# HistoPLUS Class Mapping (0 = background, 1-13 = cell types)
# Based on the HistoPLUS paper's 13 cell type taxonomy
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

# Classes to highlight in the trend plot (most biologically relevant)
KEY_CLASSES_FOR_PLOT = [
    "Cancer cells", 
    "Epithelial (Normal)", 
    "Fibroblasts", 
    "Lymphocytes",
    "Macrophages",
]

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
        logger.warning("  Set HF_TOKEN env var or create ~/.hf_token file")


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
    return parser.parse_args()


# =============================================================================
# HISTOPLUS INFERENCE
# =============================================================================

class HistoPLUSInference:
    """
    Wrapper for HistoPLUS model inference on individual patches.
    
    Note: This uses semantic segmentation (pixel-level class predictions)
    rather than full instance segmentation, which would require additional
    post-processing with HV maps and watershed.
    """
    
    def __init__(self):
        if not HISTOPLUS_AVAILABLE:
            raise RuntimeError("HistoPLUS not installed")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing HistoPLUS on {self.device}...")
        
        self.segmentor = CellViTSegmentor.from_histoplus(
            mpp=0.25,  # Target resolution (40x equivalent)
            mixed_precision=(self.device == "cuda"),
            inference_image_size=INFERENCE_SIZE,
        )
        # Move model to device manually if needed
        if hasattr(self.segmentor, 'model'):
            self.segmentor.model = self.segmentor.model.to(self.device)
            self.segmentor.model.eval()
        
        # ImageNet normalization stats (standard for ViT models)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        logger.info("✓ HistoPLUS ready")

    def preprocess(self, img_pil: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for model input.
        Upsamples 20x (224px) -> 40x equivalent (448px).
        """
        # Resize to expected input size
        img_resized = img_pil.resize(
            (INFERENCE_SIZE, INFERENCE_SIZE), 
            Image.Resampling.LANCZOS
        )
        
        # Convert to tensor and normalize
        img_arr = np.array(img_resized)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        
        return img_tensor, img_resized

    @torch.no_grad()
    def predict(self, img_pil: Image.Image) -> tuple:
        """
        Run inference on a single patch using the confirmed 'tp' key.
        """
        # 1. Resize & Preprocess
        img_tensor, img_resized = self.preprocess(img_pil)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 2. Run Model
        # This calls the inner CellViT directly, returning logits
        outputs = self.segmentor.model(img_tensor)
        
        # 3. Extract Type Logits
        # The code confirms the key is "tp"
        if "tp" not in outputs:
             # Fallback just in case, though the code guarantees "tp"
             raise KeyError(f"Expected key 'tp', found: {list(outputs.keys())}")
             
        type_logits = outputs["tp"] # Shape: (B, C, H, W)

        # 4. Generate Map
        # Argmax to get class index per pixel
        type_map = torch.argmax(type_logits, dim=1).cpu().numpy()[0]
        
        return img_resized, type_map


def get_class_pixel_fractions(type_map: np.ndarray) -> dict:
    """
    Calculate the fraction of pixels belonging to each cell class.
    
    This is a simpler and more robust metric than cell counting when
    using semantic segmentation without proper instance separation.
    
    Args:
        type_map: (H, W) array with class indices
        
    Returns:
        Dictionary mapping class names to percentage of tissue area
    """
    total_pixels = type_map.size
    fractions = {}
    
    for class_idx, class_name in HISTOPLUS_CLASSES.items():
        mask = (type_map == class_idx)
        pixel_count = mask.sum()
        fractions[class_name] = (pixel_count / total_pixels) * 100
    
    # Also compute background fraction for sanity check
    bg_mask = (type_map == 0)
    fractions["Background"] = (bg_mask.sum() / total_pixels) * 100
    
    return fractions


def get_cell_counts_approx(type_map: np.ndarray) -> dict:
    """
    Approximate cell counts using connected component analysis.
    
    WARNING: This is a rough approximation. Without proper instance
    segmentation (HV maps + watershed), touching cells of the same
    class will be counted as one.
    
    Args:
        type_map: (H, W) array with class indices
        
    Returns:
        Dictionary mapping class names to approximate cell counts
    """
    counts = {}
    
    for class_idx, class_name in HISTOPLUS_CLASSES.items():
        mask = (type_map == class_idx)
        if not np.any(mask):
            counts[class_name] = 0
            continue
        
        # Count connected components
        labeled_mask, num_components = label(mask, return_num=True)
        counts[class_name] = num_components
    
    return counts


def create_overlay(img_pil: Image.Image, type_map: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """
    Create a visualization overlay showing cell type predictions.
    
    Args:
        img_pil: Original image
        type_map: (H, W) class prediction map
        alpha: Transparency for overlay (0=invisible, 1=opaque)
        
    Returns:
        PIL Image with colored overlay
    """
    img_arr = np.array(img_pil).astype(np.float32)
    
    # Build RGB label visualization
    label_rgb = np.zeros_like(img_arr)
    
    for class_idx, class_name in HISTOPLUS_CLASSES.items():
        mask = (type_map == class_idx)
        if not np.any(mask):
            continue
        
        color = CLASS_COLORS.get(class_name, (0.5, 0.5, 0.5))
        for c in range(3):
            label_rgb[mask, c] = color[c] * 255
    
    # Alpha blend where we have predictions (non-background)
    foreground_mask = (type_map > 0)
    overlay = img_arr.copy()
    
    if np.any(foreground_mask):
        overlay[foreground_mask] = (
            (1 - alpha) * img_arr[foreground_mask] + 
            alpha * label_rgb[foreground_mask]
        )
    
    # Add white boundaries between regions for clarity
    boundaries = find_boundaries(type_map, mode='thin')
    overlay[boundaries] = [255, 255, 255]
    
    return Image.fromarray(overlay.astype(np.uint8))


# =============================================================================
# TRAJECTORY COMPUTATION
# =============================================================================

def compute_trajectory(prog_config: dict, model_name: str, num_samples: int) -> pd.DataFrame:
    """
    Compute DPT trajectory and select evenly-spaced patches.
    
    Returns:
        DataFrame with selected patches, including normalized pseudotime
    """
    # Initialize dataset
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
        # Sample patches
        patch_ids = dataset.sample_patch_ids(
            n_per_class=config.EVALUATION["n_per_class"],
            max_per_slide=config.EVALUATION["max_per_slide"],
            seed=config.EVALUATION["seed"]
        )
        
        # Load embeddings
        dataset.load_model_into_memory(model_name, patch_ids)
        cohort_df = dataset.get_cohort(patch_ids, model_name, EMBEDDING_TYPE)
        
        if cohort_df.empty:
            raise ValueError(f"No data found for {model_name}")
        
        # Convert to AnnData and compute DPT
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
        
        # Extract valid pseudotime values
        trajectory_df = adata.obs.copy()
        trajectory_df = trajectory_df[np.isfinite(trajectory_df["dpt_pseudotime"])]
        
        # CRITICAL: Normalize pseudotime to [0, 1]
        pt_min = trajectory_df["dpt_pseudotime"].min()
        pt_max = trajectory_df["dpt_pseudotime"].max()
        trajectory_df["dpt_pseudotime_norm"] = (
            (trajectory_df["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)
        )
        
        trajectory_df = trajectory_df.sort_values("dpt_pseudotime_norm")
        
        # Select evenly-spaced patches along normalized pseudotime
        target_values = np.linspace(0, 1, num_samples)
        pseudotimes = trajectory_df["dpt_pseudotime_norm"].values
        
        selected_ilocs = []
        for target in target_values:
            idx = (np.abs(pseudotimes - target)).argmin()
            selected_ilocs.append(idx)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ilocs = []
        for idx in selected_ilocs:
            if idx not in seen:
                seen.add(idx)
                unique_ilocs.append(idx)
        
        selected_df = trajectory_df.iloc[unique_ilocs].copy()
        selected_df['tau'] = result.tau
        
        return selected_df
        
    finally:
        dataset.clear_cache()


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_figure(
    patches_data: list,
    model_name: str,
    progression_name: str,
    tau: float,
    output_path: Path
):
    """
    Generate composite figure with:
    - Row 1: Original images
    - Row 2: Segmentation overlays
    - Row 3: Cell composition trends along pseudotime
    
    Args:
        patches_data: List of dicts with 'pt', 'orig_img', 'overlay_img', 'fractions', 'stage'
        model_name: Name of the foundation model
        progression_name: Name of the disease progression
        tau: Trajectory fidelity score
        output_path: Where to save the figure
    """
    n = len(patches_data)
    
    fig = plt.figure(figsize=(n * 2.5, 9))
    gs = GridSpec(3, n, height_ratios=[1, 1, 0.9], hspace=0.15, wspace=0.08)
    
    # Collect data for trend plot
    pseudotimes = []
    trends = {cls: [] for cls in KEY_CLASSES_FOR_PLOT}
    
    for i, pdata in enumerate(patches_data):
        pt = pdata['pt']
        pseudotimes.append(pt)
        
        # Row 1: Original Image
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(pdata['orig_img'])
        ax_orig.axis('off')
        
        # Title with stage info
        stage_name = pdata['stage']
        if len(stage_name) > 20:
            stage_name = stage_name[:18] + "..."
        ax_orig.set_title(f"PT: {pt:.2f}\n{stage_name}", fontsize=9, fontweight='bold')
        
        # Row 2: Segmentation Overlay
        ax_seg = fig.add_subplot(gs[1, i])
        ax_seg.imshow(pdata['overlay_img'])
        ax_seg.axis('off')
        
        # Collect fractions for trend plot
        for cls in KEY_CLASSES_FOR_PLOT:
            value = pdata['fractions'].get(cls, 0)
            trends[cls].append(value)
    
    # Row 3: Trend Plot (spans all columns)
    ax_trend = fig.add_subplot(gs[2, :])
    
    for cls in KEY_CLASSES_FOR_PLOT:
        color = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))
        y_vals = trends[cls]
        
        # Only plot if there's any signal
        if max(y_vals) > 0.1:  # At least 0.1% coverage somewhere
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
    
    # Legend below the plot
    ax_trend.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(KEY_CLASSES_FOR_PLOT), 5),
        frameon=False,
        fontsize=9
    )
    
    # Overall title
    fig.suptitle(
        f"Cellular Composition Along Trajectory: {progression_name}\n"
        f"Model: {model_name} (τ = {tau:.2f})",
        fontsize=12,
        fontweight='bold',
        y=0.98
    )
    
    # Add legend for segmentation colors
    legend_patches = []
    for cls in KEY_CLASSES_FOR_PLOT:
        color = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))
        legend_patches.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=cls)
        )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Figure saved to {output_path}")


def print_summary_table(patches_data: list):
    """Print a summary table of cell compositions."""
    print("\n" + "=" * 100)
    print("CELLULAR COMPOSITION SUMMARY")
    print("=" * 100)
    
    # Header
    header = f"{'PT':<8} | {'Stage':<25} | "
    header += " | ".join([f"{cls[:12]:<12}" for cls in KEY_CLASSES_FOR_PLOT])
    print(header)
    print("-" * 100)
    
    # Data rows
    for pdata in patches_data:
        row = f"{pdata['pt']:<8.2f} | {pdata['stage'][:25]:<25} | "
        values = [f"{pdata['fractions'].get(cls, 0):<12.1f}" for cls in KEY_CLASSES_FOR_PLOT]
        row += " | ".join(values)
        print(row)
    
    print("=" * 100)
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
    logger.info(f"   Sampling {num_samples} patches along trajectory")
    
    # Initialize HistoPLUS
    if not HISTOPLUS_AVAILABLE:
        logger.error("❌ HistoPLUS required for this analysis. Install with: pip install histoplus")
        sys.exit(1)
    
    try:
        histoplus = HistoPLUSInference()
    except Exception as e:
        logger.error(f"❌ Failed to initialize HistoPLUS: {e}")
        sys.exit(1)
    
    # Compute trajectory
    logger.info("\n📊 Computing trajectory...")
    selected_df = compute_trajectory(prog_config, target_model, num_samples)
    tau = selected_df['tau'].iloc[0]
    
    # Process patches
    logger.info(f"\n🔍 Processing {len(selected_df)} patches with HistoPLUS...")
    
    fs = fsspec.filesystem("gcs")
    patches_data = []
    
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        cls = row['class']
        slide_id = str(row['slide_id'])
        patch_id = str(row['patch_id'])
        
        # Ensure proper prefixes
        if not slide_id.startswith("slide_"):
            slide_id = f"slide_{slide_id}"
        if not patch_id.startswith("patch_"):
            patch_id = f"patch_{patch_id}"
        
        gcs_path = f"gs://{prog_config['bucket']}/{image_subdir}/train/{cls}/{slide_id}/{patch_id}.png"
        
        try:
            # Fetch image
            with fs.open(gcs_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            
            # Run inference
            img_resized, type_map = histoplus.predict(img)
            
            # Create overlay visualization
            overlay = create_overlay(img_resized, type_map)
            
            # Calculate pixel fractions (more robust than cell counts)
            fractions = get_class_pixel_fractions(type_map)
            
            patches_data.append({
                'pt': row['dpt_pseudotime_norm'],
                'orig_img': img_resized,
                'overlay_img': overlay,
                'fractions': fractions,
                'stage': cls
            })
            
            # Progress indicator
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
    
    # Generate figure
    plot_filename = f"histoplus_{target_model}_{target_progression.replace(' ', '_')}.png"
    plot_path = config.RESULTS_DIR / "plots" / plot_filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎨 Generating visualization...")
    generate_figure(
        patches_data,
        target_model,
        target_progression,
        tau,
        plot_path
    )
    
    # Summary
    print(f"\n✅ Analysis complete!")
    print(f"   Model: {target_model} (τ = {tau:.2f})")
    print(f"   Progression: {target_progression}")
    print(f"   Output: {plot_path}")
    
    # Quick interpretation
    if len(patches_data) >= 3:
        early_cancer = patches_data[0]['fractions'].get("Cancer cells", 0)
        late_cancer = patches_data[-1]['fractions'].get("Cancer cells", 0)
        
        if late_cancer > early_cancer + 5:
            print(f"\n   📈 Cancer cell fraction increases along trajectory "
                  f"({early_cancer:.1f}% → {late_cancer:.1f}%)")
        elif early_cancer > late_cancer + 5:
            print(f"\n   📉 Cancer cell fraction decreases along trajectory "
                  f"({early_cancer:.1f}% → {late_cancer:.1f}%) - unexpected?")
        else:
            print(f"\n   ➡️ Cancer cell fraction relatively stable "
                  f"({early_cancer:.1f}% → {late_cancer:.1f}%)")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# """
# Trajectory Inspector.

# Loads a specific model for a specific progression, computes the DPT manifold,
# and prints representative patches spanning the entire calculated trajectory.
# Also generates a visual strip of the patches.

# Usage:
#     python inspect_trajectory.py --model uni2 --progression CRC-Serrated
# """

# import sys
# import argparse
# import urllib.parse
# from pathlib import Path
# import logging
# import numpy as np
# import pandas as pd

# # Plotting & Image Imports
# import matplotlib
# matplotlib.use('Agg') # Safe for headless servers
# import matplotlib.pyplot as plt
# from PIL import Image
# import fsspec

# # Ensure we can import config from parent directory
# project_root = Path(__file__).resolve().parent.parent
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

# try:
#     import config
# except ImportError:
#     print("❌ Critical Error: Could not import 'config.py'.")
#     sys.exit(1)

# from data.progression_embedding_dataset import (
#     ProgressionEmbeddingDataset,
#     RegistryConfig,
# )
# from analysis.dpt import (
#     DPTConfig,
#     DPTMetric,
#     cohort_to_anndata,
#     compute_dpt,
# )

# logging.basicConfig(level=logging.INFO, format="%(message)s")
# logger = logging.getLogger(__name__)

# # Constants
# EMBEDDING_TYPE = "final_embedding"

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Inspect DPT trajectory for a specific model and progression."
#     )
#     parser.add_argument(
#         "--model", 
#         type=str, 
#         required=True, 
#         choices=config.EXPECTED_MODELS,
#         help="Name of the model to inspect (must be in config.EXPECTED_MODELS)"
#     )
#     parser.add_argument(
#         "--progression", 
#         type=str, 
#         required=True, 
#         help="Name of the progression (must be in config.PROGRESSIONS)"
#     )
#     parser.add_argument(
#         "--num_samples", 
#         type=int, 
#         default=10, 
#         help="Number of patches to sample along the trajectory"
#     )
#     return parser.parse_args()

# def plot_trajectory_images(df: pd.DataFrame, bucket: str, image_subdir: str, output_path: Path):
#     """
#     Fetches images from GCS and plots them in order of pseudotime.
#     Path format: gs://{bucket}/{image_subdir}/train/{class}/{slide_id}/{patch_id}.png
#     """
#     fs = fsspec.filesystem("gcs")
#     n = len(df)
    
#     # Create figure: dynamic width based on number of samples
#     fig, axes = plt.subplots(1, n, figsize=(max(4, n * 2.5), 4.5))
#     if n == 1: axes = [axes]
    
#     logger.info(f"🎨 Plotting {n} images from GCS...")
    
#     for i, (_, row) in enumerate(df.iterrows()):
#         ax = axes[i]
        
#         # 1. Metadata extraction
#         cls = row['class']
#         pt = row['dpt_pseudotime']
#         stage_val = row.get('stage_int', -1)
#         stage = int(stage_val) if pd.notna(stage_val) else -1
        
#         # 2. Path Construction
#         # Ensure 'slide_' and 'patch_' prefixes
#         slide_id = str(row['slide_id'])
#         patch_id = str(row['patch_id'])
        
#         if not slide_id.startswith("slide_"): slide_id = f"slide_{slide_id}"
#         if not patch_id.startswith("patch_"): patch_id = f"patch_{patch_id}"
        
#         # GCS Path (fsspec handles spaces in class names automatically)
#         gcs_path = f"gs://{bucket}/{image_subdir}/train/{cls}/{slide_id}/{patch_id}.png"
        
#         # 3. Fetch & Plot
#         try:
#             with fs.open(gcs_path, 'rb') as f:
#                 img = Image.open(f).convert("RGB")
#             ax.imshow(img)
#             ax.set_xticks([])
#             ax.set_yticks([])
#         except Exception as e:
#             logger.warning(f"Could not load image {gcs_path}: {e}")
#             ax.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center', color='red')
#             ax.axis('off')

#         # 4. Labeling
#         title_color = 'black'
#         # Optional: Color code title by stage/progression if needed
        
#         ax.set_title(
#             f"PT: {pt:.3f}\nStage: {stage}\n{cls}", 
#             fontsize=9, 
#             wrap=True,
#             color=title_color
#         )
#         # Remove box spines
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     logger.info(f"✅ Plot saved to {output_path}")

# def main():
#     args = parse_args()
    
#     target_model = args.model
#     target_progression = args.progression
#     num_samples = args.num_samples

#     # 1. Locate the specific progression config
#     prog_config = next(
#         (p for p in config.PROGRESSIONS if p["name"] == target_progression), 
#         None
#     )
    
#     if not prog_config:
#         available_progs = [p["name"] for p in config.PROGRESSIONS]
#         logger.error(f"❌ Progression '{target_progression}' not found in config.")
#         logger.error(f"   Available progressions: {available_progs}")
#         sys.exit(1)
        
#     # Get image subdir (fallback to 'imagenet' if missing for safety)
#     image_subdir = prog_config.get("image_subdir", "imagenet")

#     logger.info(f"🔍 Inspecting Trajectory: {target_model} on {target_progression}")
#     logger.info(f"   Bucket: {prog_config['bucket']} | Subdir: {image_subdir}")

#     # 2. Initialize Dataset
#     registry_config = RegistryConfig(
#         bucket=prog_config["bucket"],
#         prefix=prog_config["prefix"],
#         ordered_classes=prog_config["classes"],
#         models=[target_model], # Only load what we need
#         progression_name=target_progression,
#         scan_all_models=False
#     )
#     dataset = ProgressionEmbeddingDataset(registry_config)

#     # 3. Sample Patches
#     patch_ids = dataset.sample_patch_ids(
#         n_per_class=config.EVALUATION["n_per_class"],
#         max_per_slide=config.EVALUATION["max_per_slide"],
#         seed=config.EVALUATION["seed"]
#     )

#     # 4. Load Data & Compute Manifold
#     try:
#         # Load into memory
#         dataset.load_model_into_memory(target_model, patch_ids)
        
#         # Get Cohort DataFrame
#         cohort_df = dataset.get_cohort(patch_ids, target_model, EMBEDDING_TYPE)
        
#         if cohort_df.empty:
#             logger.error("No data found.")
#             return

#         # Convert to AnnData
#         adata = cohort_to_anndata(cohort_df, prog_config["classes"])

#         # Configure DPT (Metrics don't matter here, we just want the trajectory)
#         dpt_conf = DPTConfig(
#             n_neighbors=config.DPT["n_neighbors"],
#             n_diffusion_components=config.DPT["n_diffusion_components"],
#             metrics=set(), 
#             subsample_size=2000
#         )

#         logger.info("⚙️  Computing Diffusion Pseudotime...")
#         result = compute_dpt(adata, prog_config["root_class"], dpt_conf)

#         if "dpt_pseudotime" not in adata.obs:
#             logger.error("❌ DPT computation failed (no pseudotime generated).")
#             return

#         # 5. Extract and Sort Trajectory
#         trajectory_df = adata.obs.copy()
#         trajectory_df = trajectory_df[np.isfinite(trajectory_df["dpt_pseudotime"])]
#         trajectory_df = trajectory_df.sort_values("dpt_pseudotime")

#         if len(trajectory_df) == 0:
#             logger.error("No valid pseudotime points found.")
#             return

#         # 6. Select N Evenly Spaced Patches (By VALUE, not Rank)
#         target_values = np.linspace(0, 1, num_samples)
#         pseudotimes = trajectory_df["dpt_pseudotime"].values
#         selected_ilocs = []
        
#         for target in target_values:
#             idx = (np.abs(pseudotimes - target)).argmin()
#             selected_ilocs.append(idx)
            
#         selected_ilocs = sorted(list(set(selected_ilocs)))
#         selected_patches = trajectory_df.iloc[selected_ilocs]

#         # 7. Print Result
#         print("\n" + "="*160)
#         print(f"TRAJECTORY INSPECTION: {target_model} | {target_progression}")
#         print("="*160)
#         print(f"{'Pseudotime':<12} | {'Stage':<5} | {'Class':<30} | {'Slide ID':<20} | {'GCS URL'}")
#         print("-" * 160)

#         for _, row in selected_patches.iterrows():
#             pt = f"{row['dpt_pseudotime']:.4f}"
#             stage_val = row.get('stage_int', -1)
#             stage = str(int(stage_val)) if pd.notna(stage_val) else "?"
#             cls = row['class']
#             slide_disp = (row['slide_id'][:18] + '..') if len(row['slide_id']) > 20 else row['slide_id']

#             encoded_class = urllib.parse.quote(cls)
#             bucket_name = prog_config['bucket']
            
#             # Handle potential missing prefixes
#             slide_id_raw = str(row['slide_id'])
#             patch_id_raw = str(row['patch_id'])
            
#             slide_part = slide_id_raw if slide_id_raw.startswith("slide_") else f"slide_{slide_id_raw}"
#             patch_part = patch_id_raw if patch_id_raw.startswith("patch_") else f"patch_{patch_id_raw}"

#             # UPDATED URL to use dynamic image_subdir
#             url = f"https://storage.cloud.google.com/{bucket_name}/{image_subdir}/train/{encoded_class}/{slide_part}/{patch_part}.png"
            
#             print(f"{pt:<12} | {stage:<5} | {cls:<30} | {slide_disp:<20} | {url}")
#         print("="*160 + "\n")

#         # 8. Generate Plot
#         plot_filename = f"trajectory_{target_model}_{target_progression.replace(' ', '_')}.png"
#         plot_path = config.RESULTS_DIR / "plots" / plot_filename
#         plot_path.parent.mkdir(parents=True, exist_ok=True)
        
#         plot_trajectory_images(selected_patches, prog_config['bucket'], image_subdir, plot_path)

#     except Exception as e:
#         logger.error(f"Failed to inspect trajectory: {e}", exc_info=True)
#     finally:
#         dataset.clear_cache()

# if __name__ == "__main__":
#     main()