#!/usr/bin/env python
"""
Comprehensive Manifold Evaluation Script.

Evaluates all (progression, model, embedding_type) combinations using
Diffusion Pseudotime (DPT) and a suite of manifold quality metrics.

Usage:
    python generate_data/evaluate_dpt.py
    python generate_data/evaluate_dpt.py --n_bootstrap 50
    python generate_data/evaluate_dpt.py --output custom_output.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set, Dict, Tuple, Any

# --- Import Path Setup ---
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm
import skdim

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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Fixed scalar metrics
METRIC_KEYS = [
    "tau",
    "spectral_gap",
    "neighborhood_purity",
    "silhouette",
    "trustworthiness",
    "id_raw",
    "id_diff"
]

# Set of all metrics to calculate
ALL_METRICS = {
    DPTMetric.TAU,
    DPTMetric.SPECTRAL_GAP,
    DPTMetric.NEIGHBORHOOD_PURITY,
    DPTMetric.SILHOUETTE,
    DPTMetric.TRUSTWORTHINESS,
    DPTMetric.ID_RAW,
    DPTMetric.ID_DIFF,
    DPTMetric.PAIRWISE_AUROC  # <--- Added to set
}

ALL_EMBEDDING_TYPES = (
    config.PATCH_EMBEDDINGS +
    config.CLS_EMBEDDINGS +
    config.REGISTER_EMBEDDINGS +
    ["final_embedding"]
)

N_BOOTSTRAPS = 100

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def calc_ci(values: List[float]) -> Tuple[float, float]:
    """Helper to calculate 95% CI."""
    if not values or len(values) < 2:
        return np.nan, np.nan
    return np.percentile(values, [2.5, 97.5])


def evaluate_single_condition(
    dataset: ProgressionEmbeddingDataset,
    patch_ids: Set[str],
    model: str,
    embedding_type: str,
    ordered_classes: List[str],
    root_class: str,
    dpt_config: DPTConfig,
    n_bootstrap: int
) -> Dict[str, Any]:
    """
    Evaluates one specific (Model, EmbeddingType) pair.
    """

    # 1. Fetch Data
    try:
        cohort_df = dataset.get_cohort(
            patch_ids,
            model=model,
            embedding_type=embedding_type,
        )
    except Exception:
        return {}

    if cohort_df.empty:
        return {}

    if cohort_df.isnull().values.any():
        # ... (Data corruption check omitted for brevity, same as before) ...
        raise ValueError("Halting due to corrupted data merge.")

    adata = cohort_to_anndata(cohort_df, ordered_classes)

    # 2. Main Point Estimate
    main_result = compute_dpt(adata, root_class, dpt_config)

    # 3. Bootstrapping (Conditional)
    # We need to handle dynamic AUROC keys for bootstrapping
    # Initialize basic metrics
    boot_samples = {m: [] for m in METRIC_KEYS}

    # Initialize AUROC keys dynamically based on the main result
    # (Since we don't know the pair names until we run it once)
    auroc_keys = list(main_result.pairwise_auroc.keys())
    for k in auroc_keys:
        boot_samples[f"auroc_{k}"] = []

    if n_bootstrap > 0:
        for i in range(n_bootstrap):
            boot_ids = dataset.bootstrap_patch_ids(patch_ids, seed=i)
            boot_df = dataset.get_cohort(boot_ids, model, embedding_type)
            boot_adata = cohort_to_anndata(boot_df, ordered_classes)
            result = compute_dpt(boot_adata, root_class, dpt_config)

            if result.is_valid:
                # Store Standard Metrics
                for metric in METRIC_KEYS:
                    val = getattr(result, metric)
                    boot_samples[metric].append(val)

                # Store AUROC Metrics
                for pair_key, val in result.pairwise_auroc.items():
                    # Ensure key exists in accumulator (should match main_result)
                    full_key = f"auroc_{pair_key}"
                    if full_key in boot_samples:
                        boot_samples[full_key].append(val)

    # 4. Construct Output Row
    row = {
        "embedding_type": embedding_type,
        "n_samples": len(patch_ids),
        "bootstrap_iters": n_bootstrap
    }

    # Process Standard Metrics
    for metric in METRIC_KEYS:
        val = getattr(main_result, metric)
        row[metric] = val
        if n_bootstrap > 0:
            lower, upper = calc_ci(boot_samples[metric])
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper

    # Process AUROC Metrics
    for pair_key, val in main_result.pairwise_auroc.items():
        col_name = f"auroc_{pair_key}"
        row[col_name] = val

        if n_bootstrap > 0:
            # We use the dynamic key we created in boot_samples
            samples = boot_samples.get(col_name, [])
            lower, upper = calc_ci(samples)
            row[f"{col_name}_ci_lower"] = lower
            row[f"{col_name}_ci_upper"] = upper

    return row


def run_full_evaluation(n_bootstraps: int = N_BOOTSTRAPS) -> pd.DataFrame:

    all_results = []

    # --- LEVEL 1: PROGRESSIONS ---
    for prog_config in config.PROGRESSIONS:
        prog_name = prog_config["name"]
        ordered_classes = prog_config["classes"] # Get these here

        logger.info("=" * 60)
        logger.info(f"Processing Progression: {prog_name}")
        logger.info("=" * 60)

        registry_config = RegistryConfig(
            bucket=prog_config["bucket"],
            prefix=prog_config["prefix"],
            ordered_classes=ordered_classes,
            models=config.EXPECTED_MODELS,
            progression_name=prog_name,
            scan_all_models=False
        )
        dataset = ProgressionEmbeddingDataset(registry_config)

        # Pass ordered_classes to DPTConfig
        dpt_config = DPTConfig(
            n_neighbors=config.DPT["n_neighbors"],
            n_diffusion_components=config.DPT["n_diffusion_components"],
            metrics=ALL_METRICS,
            subsample_size=2000,
            ordered_classes=ordered_classes
        )

        patch_ids = dataset.sample_patch_ids(
            n_per_class=config.EVALUATION["n_per_class"],
            max_per_slide=config.EVALUATION["max_per_slide"],
            seed=config.EVALUATION["seed"]
        )

        # --- LEVEL 2: MODELS ---
        for model in config.EXPECTED_MODELS:
            logger.info(f"\n  Model: {model}")

            try:
                dataset.load_model_into_memory(model, patch_ids)

                # --- LEVEL 3: EMBEDDING TYPES ---
                pbar = tqdm(ALL_EMBEDDING_TYPES, desc="    Scanning layers", leave=False)

                for emb_type in pbar:
                    if emb_type == "final_embedding":
                        n_boot = n_bootstraps
                    else:
                        n_boot = 0

                    row = evaluate_single_condition(
                        dataset=dataset,
                        patch_ids=patch_ids,
                        model=model,
                        embedding_type=emb_type,
                        ordered_classes=ordered_classes,
                        root_class=prog_config["root_class"],
                        dpt_config=dpt_config,
                        n_bootstrap=n_boot
                    )

                    if row:
                        row["progression"] = prog_name
                        row["model"] = model
                        all_results.append(row)

            except Exception as e:
                logger.error(f"Failed to process model {model}: {e}")
            finally:
                dataset.clear_cache()

    return pd.DataFrame(all_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full manifold evaluation with Diffusion Pseudotime (DPT)."
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=N_BOOTSTRAPS,
        help=f"Number of bootstrap iterations for CI (default: {N_BOOTSTRAPS})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output CSV path (default: results/<timestamp>_dpt_eval/)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = run_full_evaluation(n_bootstraps=args.n_bootstrap)

    if not df.empty:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = config.get_output_dir("dpt_eval")
            output_path = output_dir / "full_manifold_evaluation.csv"

        df.to_csv(output_path, index=False)
        logger.info(f"\n Evaluation Complete. Saved to {output_path}")

        # Summary printing
        summary_cols = ["progression", "model", "tau"]
        # Try to add an AUROC column to the summary if it exists in the DF
        auroc_cols = [c for c in df.columns if "auroc" in c and "ci" not in c]
        if auroc_cols:
            summary_cols.extend(auroc_cols[:2]) # Just show first 2 pairs to keep table clean

        summary = df[df["embedding_type"] == "final_embedding"][summary_cols].sort_values(["progression", "tau"], ascending=[True, False])

        print("\nSummary (Final Embeddings):")
        print(summary.to_markdown(index=False, floatfmt=".3f"))
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()
