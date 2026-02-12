#!/usr/bin/env python
"""
Trajectory Cell Counts Analysis with HistoPLUS.

Computes diffusion pseudotime for patches and extracts cell counts using
HistoPLUS instance segmentation (proper cell counting, not pixel fractions).

Usage:
    # Test mode (10 patches)
    python generate_data/analyze_trajectory_cellcounts.py --test

    # Full run (4000 patches)
    python generate_data/analyze_trajectory_cellcounts.py

    # Custom parameters
    python generate_data/analyze_trajectory_cellcounts.py --progression CRC-Conventional --model uni2 --n_per_class 1000
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# --- Import Path Setup ---
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config
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

# HistoPLUS imports
try:
    from histoplus.helpers.segmentor import CellViTSegmentor
    from histoplus.helpers.postprocessor import CellViTPostprocessor
    HISTOPLUS_AVAILABLE = True
except ImportError:
    print("Warning: HistoPLUS not found. Install with: pip install histoplus")
    HISTOPLUS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# HistoPLUS cell type mapping (from model)
# These will be loaded from the model, but we define expected names for CSV columns
CELL_TYPE_COLUMNS = [
    "cancer_cells",
    "lymphocytes",
    "fibroblasts",
    "plasmocytes",
    "eosinophils",
    "neutrophils",
    "macrophages",
    "smooth_muscle",
    "endothelial",
    "red_blood_cells",
    "epithelial_normal",
    "mitotic_figures",
    "apoptotic_bodies",
]

# Mapping from HistoPLUS output names to our CSV column names
HISTOPLUS_TO_COLUMN = {
    "Cancer cell": "cancer_cells",
    "Lymphocytes": "lymphocytes",
    "Fibroblasts": "fibroblasts",
    "Plasmocytes": "plasmocytes",
    "Eosinophils": "eosinophils",
    "Neutrophils": "neutrophils",
    "Macrophages": "macrophages",
    "Muscle Cell": "smooth_muscle",
    "Endothelial Cell": "endothelial",
    "Red blood cell": "red_blood_cells",
    "Epithelial": "epithelial_normal",
    "Apoptotic Body": "apoptotic_bodies",
    "Mitotic Figures": "mitotic_figures",
    "Minor Stromal Cell": "minor_stromal",
}

# 20x model parameters
MPP_20X = 0.5
INFERENCE_IMAGE_SIZE_20X = 224  # Native patch size at 20x

# Local image directories (mapping bucket names to local paths)
LOCAL_IMAGE_DIRS = {
    "spider-colorectal": Path.home() / "data" / "spider-col",
    "spider-skin": Path.home() / "data" / "spider-skin",
    "spider-breast": Path.home() / "data" / "spider-breast",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CellCounts:
    """Container for cell counts from a single patch."""
    total_cells: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    mean_confidence: float = 0.0

    def __post_init__(self):
        # Initialize all cell type counts to 0
        if not self.counts:
            self.counts = {col: 0 for col in CELL_TYPE_COLUMNS}

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame row."""
        result = {
            "total_cells": self.total_cells,
            "mean_confidence": self.mean_confidence,
        }

        # Add counts
        for col in CELL_TYPE_COLUMNS:
            result[col] = self.counts.get(col, 0)

        # Add proportions
        for col in CELL_TYPE_COLUMNS:
            prop_col = f"{col}_prop"
            if self.total_cells > 0:
                result[prop_col] = self.counts.get(col, 0) / self.total_cells
            else:
                result[prop_col] = 0.0

        return result


# =============================================================================
# HISTOPLUS INFERENCE
# =============================================================================

class HistoPLUSCellCounter:
    """
    HistoPLUS wrapper that returns cell COUNTS (not pixel fractions).

    Uses the 20x model and proper postprocessing to get instance-level
    cell detections with types, centroids, and confidence scores.
    """

    def __init__(self, device: Optional[str] = None):
        if not HISTOPLUS_AVAILABLE:
            raise RuntimeError("HistoPLUS not installed. Install with: pip install histoplus")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing HistoPLUS (20x model) on {self.device}...")

        # Initialize the segmentor with 20x model
        self.segmentor = CellViTSegmentor.from_histoplus(
            mpp=MPP_20X,
            mixed_precision=(self.device == "cuda"),
            inference_image_size=INFERENCE_IMAGE_SIZE_20X,
        )

        # Move model to device
        if hasattr(self.segmentor, 'model'):
            self.segmentor.model = self.segmentor.model.to(self.device)
            self.segmentor.model.eval()

        # Get the postprocessor for instance segmentation
        self.postprocess_fn = self.segmentor.get_postprocess_fn()

        # Store the class mapping from the model
        self.class_mapping = self.segmentor.class_mapping
        logger.info(f"   Cell types: {list(self.class_mapping.values())}")

        # Normalization parameters (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        logger.info("HistoPLUS ready (20x model, instance segmentation)")

    def preprocess(self, img_pil: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for model input."""
        # Ensure correct size (should already be 224x224 for 20x patches)
        if img_pil.size != (INFERENCE_IMAGE_SIZE_20X, INFERENCE_IMAGE_SIZE_20X):
            img_pil = img_pil.resize(
                (INFERENCE_IMAGE_SIZE_20X, INFERENCE_IMAGE_SIZE_20X),
                Image.Resampling.LANCZOS
            )

        # Convert to tensor and normalize
        img_arr = np.array(img_pil)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor

    @torch.no_grad()
    def count_cells(self, img_pil: Image.Image) -> CellCounts:
        """
        Run inference and return cell COUNTS (not pixel fractions).

        Args:
            img_pil: PIL Image (224x224 RGB)

        Returns:
            CellCounts object with counts per cell type
        """
        # Preprocess
        img_tensor = self.preprocess(img_pil)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Run model forward pass
        outputs = self.segmentor.model(img_tensor)

        # The segmentor.forward() applies some processing, let's do it manually
        # to match what the postprocessor expects
        np_pred = torch.softmax(outputs["np"], dim=1)[:, 1]  # Nuclei probability
        tp_pred = torch.argmax(outputs["tp"], dim=1, keepdim=False)  # Type prediction
        hv_pred = outputs["hv"]  # Horizontal-vertical gradients

        # Convert to numpy for postprocessing
        outputs_np = {
            "np": np_pred.cpu().numpy(),
            "tp": tp_pred.cpu().numpy(),
            "hv": hv_pred.cpu().numpy(),
        }

        # Run postprocessing to get individual cell instances
        tile_predictions = self.postprocess_fn(outputs_np)

        # Should return a list with one TilePrediction (batch size = 1)
        if not tile_predictions:
            return CellCounts()

        tile_pred = tile_predictions[0]

        # Count cells by type
        counts = CellCounts()
        counts.total_cells = len(tile_pred.cell_types)

        if counts.total_cells == 0:
            return counts

        # Initialize counts
        counts.counts = {col: 0 for col in CELL_TYPE_COLUMNS}

        # Aggregate counts by cell type
        confidences = []
        for cell_type, confidence in zip(tile_pred.cell_types, tile_pred.cell_type_probabilities):
            col_name = HISTOPLUS_TO_COLUMN.get(cell_type)
            if col_name:
                counts.counts[col_name] += 1
            else:
                logger.warning(f"Unknown cell type: {cell_type}")
            confidences.append(confidence)

        counts.mean_confidence = np.mean(confidences) if confidences else 0.0

        return counts


# =============================================================================
# MAIN ANALYSIS PIPELINE
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
        logger.info("HuggingFace token loaded")
    else:
        logger.warning("No HF_TOKEN found. HistoPLUS download may fail.")


def get_progression_config(progression_name: str) -> dict:
    """Get progression config by name."""
    for prog in config.PROGRESSIONS:
        if prog["name"] == progression_name:
            return prog

    available = [p["name"] for p in config.PROGRESSIONS]
    raise ValueError(f"Progression '{progression_name}' not found. Available: {available}")


def compute_pseudotime_for_patches(
    prog_config: dict,
    model_name: str,
    patch_ids: set,
    dataset: ProgressionEmbeddingDataset,
) -> pd.DataFrame:
    """
    Compute diffusion pseudotime for all patches.

    Returns DataFrame with columns: patch_id, slide_id, class, pseudotime, pseudotime_norm
    """
    logger.info(f"Computing Diffusion Pseudotime for {len(patch_ids)} patches...")

    # Load embeddings
    dataset.load_model_into_memory(model_name, patch_ids)
    cohort_df = dataset.get_cohort(patch_ids, model_name, config.EVALUATION["embedding_type"])

    if cohort_df.empty:
        raise ValueError(f"No embeddings found for model {model_name}")

    # Convert to AnnData and compute DPT
    adata = cohort_to_anndata(cohort_df, prog_config["classes"])

    dpt_config = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=config.DPT["n_diffusion_components"],
        metrics={DPTMetric.TAU},
        subsample_size=None  # Use all patches
    )

    result = compute_dpt(adata, prog_config["root_class"], dpt_config)
    logger.info(f"   Trajectory Fidelity: tau = {result.tau:.3f}")

    # Extract pseudotime values
    trajectory_df = adata.obs.copy()
    trajectory_df = trajectory_df[np.isfinite(trajectory_df["dpt_pseudotime"])]

    # Normalize pseudotime to [0, 1]
    pt_min = trajectory_df["dpt_pseudotime"].min()
    pt_max = trajectory_df["dpt_pseudotime"].max()
    trajectory_df["pseudotime_norm"] = (
        (trajectory_df["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)
    )
    trajectory_df = trajectory_df.rename(columns={"dpt_pseudotime": "pseudotime"})

    # The index of adata.obs is the patch_id - reset it to make it a column
    # First, check if patch_id already exists as a column
    if "patch_id" in trajectory_df.columns:
        # patch_id is already a column, just reset index without keeping it
        trajectory_df = trajectory_df.reset_index(drop=True)
    else:
        # patch_id is in the index, reset it to make it a column
        trajectory_df = trajectory_df.reset_index()
        # The index name might be None or 'index', rename to patch_id
        if trajectory_df.columns[0] == "index" or trajectory_df.columns[0] is None:
            trajectory_df = trajectory_df.rename(columns={trajectory_df.columns[0]: "patch_id"})

    # Debug: log columns to understand structure
    logger.info(f"   Trajectory columns: {list(trajectory_df.columns)}")
    logger.info(f"   First row sample: {trajectory_df.iloc[0].to_dict()}")

    logger.info(f"Pseudotime computed for {len(trajectory_df)} patches")

    return trajectory_df


def run_analysis(
    progression_name: str,
    model_name: str,
    n_per_class: int,
    output_path: Path,
    test_mode: bool = False,
    seed: int = 42,
    data_dir: Optional[Path] = None,
):
    """
    Main analysis pipeline.

    1. Sample patches (n_per_class from each disease stage)
    2. Compute pseudotime for all patches
    3. Run HistoPLUS on each patch to get cell counts
    4. Save results to CSV
    """
    start_time = time.time()

    # Get progression config
    prog_config = get_progression_config(progression_name)
    logger.info(f"Analyzing: {progression_name}")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Classes: {prog_config['classes']}")

    # Adjust for test mode
    if test_mode:
        n_per_class = 3  # 3 per class = ~10-12 patches total
        logger.info(f"   TEST MODE: Using {n_per_class} patches per class")
    else:
        logger.info(f"   Patches per class: {n_per_class}")

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

    # Sample patch IDs
    patch_ids = dataset.sample_patch_ids(
        n_per_class=n_per_class,
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=seed
    )
    logger.info(f"   Sampled {len(patch_ids)} patches")

    try:
        # Step 1: Compute pseudotime
        trajectory_df = compute_pseudotime_for_patches(
            prog_config, model_name, patch_ids, dataset
        )

        # Step 2: Initialize HistoPLUS
        cell_counter = HistoPLUSCellCounter()

        # Step 3: Process each patch
        logger.info(f"\nRunning HistoPLUS on {len(trajectory_df)} patches...")

        # Get local image directory
        if data_dir is not None:
            local_base = Path(data_dir)
        else:
            local_base = LOCAL_IMAGE_DIRS.get(prog_config["bucket"])
            if local_base is None:
                raise ValueError(f"No local directory configured for bucket: {prog_config['bucket']}")

        if not local_base.exists():
            raise FileNotFoundError(f"Local image directory not found: {local_base}")

        logger.info(f"   Using local images from: {local_base}")

        # Debug: Check which class folders exist
        train_dir = local_base / "train"
        if train_dir.exists():
            available_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
            expected_classes = prog_config["classes"]
            logger.info(f"   Expected classes: {expected_classes}")
            logger.info(f"   Available folders: {available_classes}")

            # Check for mismatches
            missing = set(expected_classes) - set(available_classes)
            if missing:
                logger.warning(f"   Missing class folders: {missing}")

        results = []
        failed_patches = 0

        for idx, row in tqdm(trajectory_df.iterrows(), total=len(trajectory_df), desc="Processing patches"):
            # Extract values as scalars (not Series)
            patch_id = str(row["patch_id"]) if "patch_id" in row.index else str(idx)
            slide_id = str(row["slide_id"]) if "slide_id" in row.index else "unknown"
            class_label = str(row["class"]) if "class" in row.index else "unknown"

            # Format IDs for file path
            if not slide_id.startswith("slide_"):
                slide_id_fmt = f"slide_{slide_id}"
            else:
                slide_id_fmt = slide_id

            if not patch_id.startswith("patch_"):
                patch_id_fmt = f"patch_{patch_id}"
            else:
                patch_id_fmt = patch_id

            # Construct local file path
            # Structure: data/spider-col/train/{class_label}/{slide_id}/{patch_id}.png
            local_path = local_base / "train" / class_label / slide_id_fmt / f"{patch_id_fmt}.png"

            try:
                # Load image from local file
                if not local_path.exists():
                    raise FileNotFoundError(f"Image not found: {local_path}")

                img = Image.open(local_path).convert("RGB")

                # Get cell counts
                cell_counts = cell_counter.count_cells(img)

                # Build result row
                result_row = {
                    "slide_id": slide_id,
                    "patch_id": patch_id,
                    "class_label": class_label,
                    "pseudotime": float(row["pseudotime"]),
                    "pseudotime_norm": float(row["pseudotime_norm"]),
                }
                result_row.update(cell_counts.to_dict())

                results.append(result_row)

            except Exception as e:
                failed_patches += 1
                if failed_patches <= 5:  # Only log first 5 failures
                    logger.warning(f"   Failed: {patch_id_fmt} in {slide_id_fmt} - {e}")
                elif failed_patches == 6:
                    logger.warning("   ... (suppressing further warnings)")
                continue

        if failed_patches > 0:
            logger.warning(f"   Total failed patches: {failed_patches}/{len(trajectory_df)}")

        # Step 4: Create DataFrame and save
        results_df = pd.DataFrame(results)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV (even if empty, for debugging)
        results_df.to_csv(output_path, index=False)

        elapsed = time.time() - start_time
        logger.info(f"\nAnalysis complete!")
        logger.info(f"   Processed: {len(results_df)} patches")
        logger.info(f"   Time: {elapsed/60:.1f} minutes")
        logger.info(f"   Output: {output_path}")

        # Print summary statistics (only if we have results)
        if len(results_df) > 0:
            print("\n" + "=" * 60)
            print("SUMMARY STATISTICS")
            print("=" * 60)
            print(f"\nPatches by class:")
            print(results_df.groupby("class_label").size().to_string())

            print(f"\nCell counts (mean +/- std):")
            print(f"  Total cells/patch: {results_df['total_cells'].mean():.1f} +/- {results_df['total_cells'].std():.1f}")

            for col in ["cancer_cells", "lymphocytes", "fibroblasts", "epithelial_normal"]:
                if col in results_df.columns:
                    print(f"  {col}: {results_df[col].mean():.1f} +/- {results_df[col].std():.1f}")

            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("NO PATCHES PROCESSED SUCCESSFULLY")
            print("=" * 60)
            print("\nPlease check:")
            print("  1. Image directory exists and has correct structure")
            print("  2. Class names in config match folder names exactly")
            print(f"  3. Expected structure: {local_base}/train/<class>/<slide_id>/<patch_id>.png")
            print("=" * 60)

        return results_df

    finally:
        dataset.clear_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze disease trajectory with HistoPLUS cell counting."
    )
    parser.add_argument(
        "--progression",
        type=str,
        default="CRC-Conventional",
        help="Disease progression to analyze (default: CRC-Conventional)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="uni2",
        choices=config.EXPECTED_MODELS,
        help="Foundation model for pseudotime (default: uni2)"
    )
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=config.EVALUATION["n_per_class"],
        help=f"Patches per class (default: {config.EVALUATION['n_per_class']})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/<timestamp>_cellcounts/<progression>_<model>_cellcounts.csv)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process ~10 patches"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.EVALUATION["seed"],
        help=f"Random seed (default: {config.EVALUATION['seed']})"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override local image directory (default: ~/data/<bucket-name>)"
    )

    return parser.parse_args()


def main():
    load_hf_token()
    args = parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = config.get_output_dir("cellcounts")
        prog_slug = args.progression.lower().replace("-", "_").replace(" ", "_")
        filename = f"{prog_slug}_{args.model}_{args.n_per_class}_cellcounts.csv"
        if args.test:
            filename = f"{prog_slug}_{args.model}_cellcounts_TEST.csv"
        output_path = output_dir / filename

    # Run analysis
    run_analysis(
        progression_name=args.progression,
        model_name=args.model,
        n_per_class=args.n_per_class,
        output_path=output_path,
        test_mode=args.test,
        seed=args.seed,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )


if __name__ == "__main__":
    main()
