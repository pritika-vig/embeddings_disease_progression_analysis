#!/usr/bin/env python
"""
Stage Permutation Specificity Test (Exact).

Objective:
    Verify that the high Kendall's Tau is specific to the *biological* ordering
    by comparing it against ALL possible permutations of the stage labels.

    Since N_classes is small (3 or 4), N! is small (6 or 24).
    We can run an Exact Test.

Usage:
    python generate_data/evaluate_stage_permutations.py
    python generate_data/evaluate_stage_permutations.py --progressions BDC SCC
    python generate_data/evaluate_stage_permutations.py --test --progressions BDC
"""

import argparse
import sys
import logging
import math
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
OUTPUT_FILENAME = "stage_permutation_specificity.csv"
EMBEDDING_TYPE = "final_embedding"

def get_all_permutations(classes):
    """
    Generates ALL unique permutations of the class list.
    """
    # itertools.permutations returns tuples
    return list(itertools.permutations(classes))

def run_permutation_test(
    eval_config: dict = None,
    progressions: list = None,
    output_path: Path = None,
):
    if eval_config is None:
        eval_config = config.EVALUATION

    n_per_class = eval_config["n_per_class"]
    max_per_slide = eval_config["max_per_slide"]

    all_results = []

    if output_path is None:
        output_dir = config.get_output_dir()
        output_path = output_dir / OUTPUT_FILENAME
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Exact Stage Permutation Test (All Permutations)")
    logger.info(f"Output will be saved to: {output_path}")

    # --- LEVEL 1: PROGRESSIONS ---
    for prog_config in config.get_progressions(progressions):
        prog_name = prog_config["name"]
        canonical_classes = prog_config["classes"]

        # Calculate expected permutations
        n_perms = math.factorial(len(canonical_classes))

        logger.info("=" * 60)
        logger.info(f"Processing {prog_name} ({len(canonical_classes)} classes -> {n_perms} permutations)")

        # Initialize Dataset
        registry_config = RegistryConfig(
            bucket=prog_config["bucket"],
            prefix=prog_config["prefix"],
            ordered_classes=canonical_classes,
            models=config.EXPECTED_MODELS,
            progression_name=prog_name,
            scan_all_models=False
        )
        dataset = ProgressionEmbeddingDataset(registry_config)

        # DPT Config
        dpt_config = DPTConfig(
            n_neighbors=config.DPT["n_neighbors"],
            n_diffusion_components=config.DPT["n_diffusion_components"],
            metrics=set(), # Manual tau calc
            subsample_size=2000
        )

        # Sample Patches
        patch_ids = dataset.sample_patch_ids(
            n_per_class=n_per_class,
            max_per_slide=max_per_slide,
            seed=config.EVALUATION["seed"]
        )

        # Generate ALL Permutations
        all_perms = get_all_permutations(canonical_classes)
        canonical_tuple = tuple(canonical_classes)

        # --- LEVEL 2: MODELS ---
        for model in config.EXPECTED_MODELS:
            try:
                # Load Model
                dataset.load_model_into_memory(model, patch_ids)

                # Fetch Data
                cohort_df = dataset.get_cohort(patch_ids, model, EMBEDDING_TYPE)
                if cohort_df.empty:
                    continue

                # 1. Compute Fixed Manifold ONCE
                adata = cohort_to_anndata(cohort_df, canonical_classes)
                dpt_result = compute_dpt(adata, prog_config["root_class"], dpt_config)

                if "dpt_pseudotime" not in adata.obs:
                    logger.warning(f"   Skipping {model}: DPT failed.")
                    continue

                # Prepare Vectors
                valid_mask = np.isfinite(adata.obs["dpt_pseudotime"])
                fixed_pseudotime = adata.obs["dpt_pseudotime"][valid_mask].values
                raw_classes = adata.obs["class"][valid_mask].values

                # 2. Iterate through ALL permutations
                for i, perm_tuple in enumerate(all_perms):
                    perm_list = list(perm_tuple)

                    # Check if this is the canonical order
                    is_canonical = (perm_tuple == canonical_tuple)
                    type_label = "canonical" if is_canonical else "permuted"

                    # Create Map: {ClassA: 0, ClassB: 1 ...}
                    perm_map = {c: idx for idx, c in enumerate(perm_list)}

                    # Map raw classes to NEW integers
                    perm_stages = np.array([perm_map[c] for c in raw_classes])

                    # Calculate Tau
                    tau, _ = kendalltau(fixed_pseudotime, perm_stages)

                    all_results.append({
                        "progression": prog_name,
                        "model": model,
                        "type": type_label,
                        "perm_id": i,
                        "order_str": " -> ".join(perm_list),
                        "tau": tau
                    })

                logger.info(f"   {model:<10} | Computed {n_perms} permutations.")

            except Exception as e:
                logger.error(f"Failed model {model}: {e}")
            finally:
                dataset.clear_cache()

    # Save Results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        logger.info(f"\nExact Permutation Analysis Complete. Saved to {output_path}")
    else:
        logger.warning("No results generated.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact stage permutation specificity test."
    )
    parser.add_argument(
        "--progressions", type=str, nargs="+", default=None,
        help="Progression names to include (default: all). Choices: SCC, CRC-Conventional, CRC-Serrated, BDC"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: minimal samples, output to results/test/"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.test:
        output_dir = config.RESULTS_DIR / "test"
        output_dir.mkdir(parents=True, exist_ok=True)

        run_permutation_test(
            eval_config=config.TEST_EVALUATION,
            progressions=args.progressions,
            output_path=output_dir / OUTPUT_FILENAME,
        )
    else:
        run_permutation_test(
            progressions=args.progressions,
        )
