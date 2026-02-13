#!/usr/bin/env python
"""
Null Distribution Generator.

Methodology:
1. Load Embeddings & Construct Manifold (DPT) on REAL data.
2. Fix the calculated Pseudotime coordinates.
3. Permute (shuffle) the Class Labels N times.
4. Calculate Kendall's Tau for each permutation.

Usage:
    python generate_data/generate_null_distribution.py
    python generate_data/generate_null_distribution.py --n_permutations 50
    python generate_data/generate_null_distribution.py --test --progressions BDC
"""

import argparse
import logging
import sys
from pathlib import Path

# --- Import Path Setup ---
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm

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

# Configuration
EMBEDDING_TARGET = "final_embedding"


def run_null_evaluation(
    eval_config: dict = None,
    progressions: list = None,
    output_path: Path = None,
):
    if eval_config is None:
        eval_config = config.EVALUATION

    n_permutations = eval_config["n_null_permutations"]
    n_per_class = eval_config["n_per_class"]
    max_per_slide = eval_config["max_per_slide"]

    if output_path is None:
        output_dir = config.get_output_dir()
        output_path = output_dir / "null_manifold_evaluation.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output will be saved to: {output_path}")

    all_null_results = []

    # --- LEVEL 1: PROGRESSIONS ---
    for prog_config in config.get_progressions(progressions):
        prog_name = prog_config["name"]
        logger.info("=" * 60)
        logger.info(f"Processing Nulls for: {prog_name}")

        # Initialize Dataset
        registry_config = RegistryConfig(
            bucket=prog_config["bucket"],
            prefix=prog_config["prefix"],
            ordered_classes=prog_config["classes"],
            models=config.EXPECTED_MODELS,
            progression_name=prog_name,
            scan_all_models=False
        )
        dataset = ProgressionEmbeddingDataset(registry_config)

        # Standard DPT Config
        dpt_config = DPTConfig(
            n_neighbors=config.DPT["n_neighbors"],
            n_diffusion_components=config.DPT["n_diffusion_components"],
            metrics={DPTMetric.TAU},
            subsample_size=2000
        )

        # Sample Patches
        patch_ids = dataset.sample_patch_ids(
            n_per_class=n_per_class,
            max_per_slide=max_per_slide,
            seed=config.EVALUATION["seed"]
        )

        # --- LEVEL 2: MODELS ---
        for model in config.EXPECTED_MODELS:
            try:
                # Load Model
                dataset.load_model_into_memory(model, patch_ids)

                # Fetch Cohort
                cohort_df = dataset.get_cohort(
                    patch_ids,
                    model=model,
                    embedding_type=EMBEDDING_TARGET
                )

                if cohort_df.empty:
                    continue

                # 1. Compute Manifold ONCE
                adata = cohort_to_anndata(cohort_df, prog_config["classes"])
                dpt_result = compute_dpt(adata, prog_config["root_class"], dpt_config)

                if not dpt_result.is_valid:
                    logger.warning(f"  Skipping {model}: DPT failed on real data.")
                    continue

                # 2. Extract Vectors for Permutation
                valid_mask = np.isfinite(adata.obs["dpt_pseudotime"])
                fixed_pseudotime = adata.obs["dpt_pseudotime"][valid_mask].values
                original_labels = adata.obs["stage_int"][valid_mask].values

                # 3. Bootstrap Shuffle Loop
                null_taus = []
                rng = np.random.default_rng(seed=42)

                for _ in range(n_permutations):
                    shuffled_labels = rng.permutation(original_labels)
                    tau, _ = kendalltau(fixed_pseudotime, shuffled_labels)
                    null_taus.append(tau)

                # 4. Aggregate Results
                lower, upper = np.percentile(null_taus, [2.5, 97.5])

                all_null_results.append({
                    "progression": prog_name,
                    "model": model,
                    "embedding_type": "shuffled_null",
                    "tau": np.mean(null_taus),
                    "tau_ci_lower": lower,
                    "tau_ci_upper": upper,
                    "n_permutations": n_permutations
                })

                logger.info(f"  {model}: Null Tau = {np.mean(null_taus):.3f} (CI: {lower:.3f}, {upper:.3f})")

            except Exception as e:
                logger.error(f"Failed model {model}: {e}")
            finally:
                dataset.clear_cache()

    # Save Results
    if all_null_results:
        df_null = pd.DataFrame(all_null_results)
        df_null.to_csv(output_path, index=False)
        logger.info(f"\nNull Evaluation Complete. Saved to {output_path}")
    else:
        logger.warning("No null results generated.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate null distribution for DPT evaluation."
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

        run_null_evaluation(
            eval_config=config.TEST_EVALUATION,
            progressions=args.progressions,
            output_path=output_dir / "null_manifold_evaluation.csv",
        )
    else:
        run_null_evaluation(
            progressions=args.progressions,
        )
