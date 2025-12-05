#!/usr/bin/env python
"""Evaluate embedding models using Diffusion Pseudotime and Manifold Metrics."""

import logging
from typing import List, Set, Dict, Tuple

import numpy as np
import pandas as pd
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

# Metrics configuration
METRIC_KEYS = [
    "tau", 
    "spectral_gap", 
    "neighborhood_purity", 
    "trustworthiness", 
    "id_raw", 
    "id_diff"
]

ALL_METRICS = {
    DPTMetric.TAU,
    DPTMetric.SPECTRAL_GAP,
    DPTMetric.NEIGHBORHOOD_PURITY,
    DPTMetric.TRUSTWORTHINESS,
    DPTMetric.ID_RAW,
    DPTMetric.ID_DIFF
}

def get_progression_config(name: str) -> dict:
    for prog in config.PROGRESSIONS:
        if prog["name"] == name:
            return prog
    raise ValueError(f"Progression '{name}' not found in config.PROGRESSIONS")


def calc_ci(values: List[float]) -> Tuple[float, float]:
    """Helper to calculate 95% CI (2.5th and 97.5th percentiles)."""
    if not values or len(values) < 2:
        return np.nan, np.nan
    return np.percentile(values, [2.5, 97.5])


def evaluate_model(
    dataset: ProgressionEmbeddingDataset,
    patch_ids: Set[str],
    model: str,
    ordered_classes: List[str],
    root_class: str,
    dpt_config: DPTConfig,
) -> dict:
    """Evaluate a single model with bootstrapped confidence intervals for all metrics."""
    
    embedding_type = config.EVALUATION.get("embedding_type", "cls")
    
    # ---------------------------------------------------------
    # 1. Main Run on Full Sample (Point Estimate)
    # ---------------------------------------------------------
    # Note: Since we pre-loaded memory in run_evaluation, this is FAST (RAM only).
    cohort_df = dataset.get_cohort(
        patch_ids, 
        model=model, 
        embedding_type=embedding_type,
    )
    adata = cohort_to_anndata(cohort_df, ordered_classes)
    main_result = compute_dpt(adata, root_class, dpt_config)
    
    logger.info(
        "  Tau: %.3f | Purity: %.3f | ID(Diff): %.1f | Trust: %.3f", 
        main_result.tau, 
        main_result.neighborhood_purity,
        main_result.id_diff,
        main_result.trustworthiness
    )
    
    # ---------------------------------------------------------
    # 2. Bootstrapping (Uncertainty Quantification)
    # ---------------------------------------------------------
    boot_samples = {m: [] for m in METRIC_KEYS}
    n_boot = config.EVALUATION.get("n_bootstrap", 0)

    if n_boot > 0:
        for i in tqdm(range(n_boot), desc="  Bootstrap", leave=False):
            # Resample IDs
            boot_ids = dataset.bootstrap_patch_ids(patch_ids, seed=i)
            
            # Fetch data (FAST: Hits the memory cache)
            boot_df = dataset.get_cohort(
                boot_ids, 
                model=model, 
                embedding_type=embedding_type,
            )
            boot_adata = cohort_to_anndata(boot_df, ordered_classes)
            result = compute_dpt(boot_adata, root_class, dpt_config)
            
            if result.is_valid:
                for metric in METRIC_KEYS:
                    val = getattr(result, metric)
                    boot_samples[metric].append(val)
    
    # ---------------------------------------------------------
    # 3. Construct Output
    # ---------------------------------------------------------
    output = {
        "model": model,
        "n_samples": len(patch_ids)
    }

    for metric in METRIC_KEYS:
        output[metric] = getattr(main_result, metric)
        lower, upper = calc_ci(boot_samples[metric])
        output[f"{metric}_ci_lower"] = lower
        output[f"{metric}_ci_upper"] = upper

    return output


def run_evaluation(progression_name: str) -> pd.DataFrame:
    """Run evaluation for a progression across all models."""
    
    logger.info("=" * 60)
    logger.info("  Progression: %s", progression_name)
    logger.info("=" * 60)
    
    prog = get_progression_config(progression_name)
    ordered_classes = prog["classes"]
    root_class = prog["root_class"]
    
    logger.info("Root: %s", root_class)
    logger.info("Stages: %s", ordered_classes)
    
    # 1. Initialize Dataset (Golden Record Mode)
    registry_config = RegistryConfig(
        bucket=prog["bucket"],
        prefix=prog["prefix"],
        ordered_classes=ordered_classes,
        models=config.EXPECTED_MODELS,
        progression_name=progression_name,
        scan_all_models=False  # <--- NEW: Use Fast Mode
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    
    # 2. Configure DPT
    dpt_config = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=config.DPT["n_diffusion_components"],
        metrics=ALL_METRICS, 
        subsample_size=2000 
    )
    
    # 3. Sample Patches
    logger.info("\nSampling patches...")
    patch_ids = dataset.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=config.EVALUATION["seed"],
    )
    
    results = []
    for model in config.EXPECTED_MODELS:
        logger.info("\n--- %s ---", model.upper())
        try:
            # 4. OPTIMIZATION: Load Model into RAM
            # This downloads the file ONCE. All bootstrap iterations will use this.
            dataset.load_model_into_memory(model, patch_ids)
            
            result = evaluate_model(
                dataset=dataset,
                patch_ids=patch_ids,
                model=model,
                ordered_classes=ordered_classes,
                root_class=root_class,
                dpt_config=dpt_config,
            )
            result["progression"] = progression_name
            results.append(result)
            
        except Exception as e:
            logger.error("Failed: %s", e, exc_info=True)
        finally:
            # 5. CLEANUP: Free RAM
            dataset.clear_cache()
    
    return pd.DataFrame(results)


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_dpt.py <progression_name>")
        print(f"Available: {[p['name'] for p in config.PROGRESSIONS]}")
        sys.exit(1)
    
    progression_name = sys.argv[1]
    results_df = run_evaluation(progression_name)
    
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS (Sorted by Tau)")
    logger.info("=" * 80)
    
    if not results_df.empty:
        display_cols = ["model", "tau", "spectral_gap", "neighborhood_purity", "trustworthiness", "id_diff"]
        print(results_df.sort_values("tau", ascending=False)[display_cols].to_markdown(index=False, floatfmt=".4f"))
        
        output_path = f"results_dpt_{progression_name}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info("\nFull results with CIs saved to %s", output_path)


if __name__ == "__main__":
    main()