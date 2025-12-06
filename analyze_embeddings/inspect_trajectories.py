#!/usr/bin/env python
"""
Trajectory Inspector.

Loads a specific model for a specific progression, computes the DPT manifold,
and prints representative patches spanning the entire calculated trajectory.
Also generates a visual strip of the patches.

Usage:
    python inspect_trajectory.py --model uni2 --progression CRC-Serrated
"""

import sys
import argparse
import urllib.parse
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Plotting & Image Imports
import matplotlib
matplotlib.use('Agg') # Safe for headless servers
import matplotlib.pyplot as plt
from PIL import Image
import fsspec

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

# Constants
EMBEDDING_TYPE = "final_embedding"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect DPT trajectory for a specific model and progression."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=config.EXPECTED_MODELS,
        help="Name of the model to inspect (must be in config.EXPECTED_MODELS)"
    )
    parser.add_argument(
        "--progression", 
        type=str, 
        required=True, 
        help="Name of the progression (must be in config.PROGRESSIONS)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10, 
        help="Number of patches to sample along the trajectory"
    )
    return parser.parse_args()

def plot_trajectory_images(df: pd.DataFrame, bucket: str, image_subdir: str, output_path: Path):
    """
    Fetches images from GCS and plots them in order of pseudotime.
    Path format: gs://{bucket}/{image_subdir}/train/{class}/{slide_id}/{patch_id}.png
    """
    fs = fsspec.filesystem("gcs")
    n = len(df)
    
    # Create figure: dynamic width based on number of samples
    fig, axes = plt.subplots(1, n, figsize=(max(4, n * 2.5), 4.5))
    if n == 1: axes = [axes]
    
    logger.info(f"🎨 Plotting {n} images from GCS...")
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        
        # 1. Metadata extraction
        cls = row['class']
        pt = row['dpt_pseudotime']
        stage_val = row.get('stage_int', -1)
        stage = int(stage_val) if pd.notna(stage_val) else -1
        
        # 2. Path Construction
        # Ensure 'slide_' and 'patch_' prefixes
        slide_id = str(row['slide_id'])
        patch_id = str(row['patch_id'])
        
        if not slide_id.startswith("slide_"): slide_id = f"slide_{slide_id}"
        if not patch_id.startswith("patch_"): patch_id = f"patch_{patch_id}"
        
        # GCS Path (fsspec handles spaces in class names automatically)
        gcs_path = f"gs://{bucket}/{image_subdir}/train/{cls}/{slide_id}/{patch_id}.png"
        
        # 3. Fetch & Plot
        try:
            with fs.open(gcs_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            logger.warning(f"Could not load image {gcs_path}: {e}")
            ax.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center', color='red')
            ax.axis('off')

        # 4. Labeling
        title_color = 'black'
        # Optional: Color code title by stage/progression if needed
        
        ax.set_title(
            f"PT: {pt:.3f}\nStage: {stage}\n{cls}", 
            fontsize=9, 
            wrap=True,
            color=title_color
        )
        # Remove box spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Plot saved to {output_path}")

def main():
    args = parse_args()
    
    target_model = args.model
    target_progression = args.progression
    num_samples = args.num_samples

    # 1. Locate the specific progression config
    prog_config = next(
        (p for p in config.PROGRESSIONS if p["name"] == target_progression), 
        None
    )
    
    if not prog_config:
        available_progs = [p["name"] for p in config.PROGRESSIONS]
        logger.error(f"❌ Progression '{target_progression}' not found in config.")
        logger.error(f"   Available progressions: {available_progs}")
        sys.exit(1)
        
    # Get image subdir (fallback to 'imagenet' if missing for safety)
    image_subdir = prog_config.get("image_subdir", "imagenet")

    logger.info(f"🔍 Inspecting Trajectory: {target_model} on {target_progression}")
    logger.info(f"   Bucket: {prog_config['bucket']} | Subdir: {image_subdir}")

    # 2. Initialize Dataset
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=[target_model], # Only load what we need
        progression_name=target_progression,
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)

    # 3. Sample Patches
    patch_ids = dataset.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=config.EVALUATION["seed"]
    )

    # 4. Load Data & Compute Manifold
    try:
        # Load into memory
        dataset.load_model_into_memory(target_model, patch_ids)
        
        # Get Cohort DataFrame
        cohort_df = dataset.get_cohort(patch_ids, target_model, EMBEDDING_TYPE)
        
        if cohort_df.empty:
            logger.error("No data found.")
            return

        # Convert to AnnData
        adata = cohort_to_anndata(cohort_df, prog_config["classes"])

        # Configure DPT (Metrics don't matter here, we just want the trajectory)
        dpt_conf = DPTConfig(
            n_neighbors=config.DPT["n_neighbors"],
            n_diffusion_components=config.DPT["n_diffusion_components"],
            metrics=set(), 
            subsample_size=2000
        )

        logger.info("⚙️  Computing Diffusion Pseudotime...")
        result = compute_dpt(adata, prog_config["root_class"], dpt_conf)

        if "dpt_pseudotime" not in adata.obs:
            logger.error("❌ DPT computation failed (no pseudotime generated).")
            return

        # 5. Extract and Sort Trajectory
        trajectory_df = adata.obs.copy()
        trajectory_df = trajectory_df[np.isfinite(trajectory_df["dpt_pseudotime"])]
        trajectory_df = trajectory_df.sort_values("dpt_pseudotime")

        if len(trajectory_df) == 0:
            logger.error("No valid pseudotime points found.")
            return

        # 6. Select N Evenly Spaced Patches (By VALUE, not Rank)
        target_values = np.linspace(0, 1, num_samples)
        pseudotimes = trajectory_df["dpt_pseudotime"].values
        selected_ilocs = []
        
        for target in target_values:
            idx = (np.abs(pseudotimes - target)).argmin()
            selected_ilocs.append(idx)
            
        selected_ilocs = sorted(list(set(selected_ilocs)))
        selected_patches = trajectory_df.iloc[selected_ilocs]

        # 7. Print Result
        print("\n" + "="*160)
        print(f"TRAJECTORY INSPECTION: {target_model} | {target_progression}")
        print("="*160)
        print(f"{'Pseudotime':<12} | {'Stage':<5} | {'Class':<30} | {'Slide ID':<20} | {'GCS URL'}")
        print("-" * 160)

        for _, row in selected_patches.iterrows():
            pt = f"{row['dpt_pseudotime']:.4f}"
            stage_val = row.get('stage_int', -1)
            stage = str(int(stage_val)) if pd.notna(stage_val) else "?"
            cls = row['class']
            slide_disp = (row['slide_id'][:18] + '..') if len(row['slide_id']) > 20 else row['slide_id']

            encoded_class = urllib.parse.quote(cls)
            bucket_name = prog_config['bucket']
            
            # Handle potential missing prefixes
            slide_id_raw = str(row['slide_id'])
            patch_id_raw = str(row['patch_id'])
            
            slide_part = slide_id_raw if slide_id_raw.startswith("slide_") else f"slide_{slide_id_raw}"
            patch_part = patch_id_raw if patch_id_raw.startswith("patch_") else f"patch_{patch_id_raw}"

            # UPDATED URL to use dynamic image_subdir
            url = f"https://storage.cloud.google.com/{bucket_name}/{image_subdir}/train/{encoded_class}/{slide_part}/{patch_part}.png"
            
            print(f"{pt:<12} | {stage:<5} | {cls:<30} | {slide_disp:<20} | {url}")
        print("="*160 + "\n")

        # 8. Generate Plot
        plot_filename = f"trajectory_{target_model}_{target_progression.replace(' ', '_')}.png"
        plot_path = config.RESULTS_DIR / "plots" / plot_filename
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_trajectory_images(selected_patches, prog_config['bucket'], image_subdir, plot_path)

    except Exception as e:
        logger.error(f"Failed to inspect trajectory: {e}", exc_info=True)
    finally:
        dataset.clear_cache()

if __name__ == "__main__":
    main()