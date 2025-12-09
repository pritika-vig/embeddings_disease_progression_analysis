#!/usr/bin/env python
"""
Comprehensive Manifold Evaluation Script.

Scope:
- All Progressions (Diseases)
- All Models
- All Embedding Types (Patch, CLS, Register, Final)
- Bootstrapping: ON for 'final_embedding' (n=50), OFF for others.
"""

import logging
import sys
from typing import List, Set, Dict, Tuple, Any

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

# Metrics to capture in CSV
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
    DPTMetric.ID_DIFF
}

# Aggregate all embedding types defined in config into one iteration list
# Order matters: We'll iterate these inner loops
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
    Handles In-Memory filtering and Conditional Bootstrapping.
    """
    
    # 1. Fetch Data
    try:
        cohort_df = dataset.get_cohort(
            patch_ids, 
            model=model, 
            embedding_type=embedding_type,
        )
    except Exception:
        # Graceful fallback if embedding type doesn't exist for model (e.g. ResNet Registers)
        return {}

    if cohort_df.empty:
        return {}

    if cohort_df.isnull().values.any():
        print(f"\n[CRITICAL] Data Corruption Detected in {model} / {embedding_type}")
        
        # Filter for the bad rows
        bad_rows = cohort_df[cohort_df.isnull().any(axis=1)]
        
        for idx, row in bad_rows.iterrows():
            pid = row.get('patch_id', 'UNKNOWN')
            cls = row.get('class', 'UNKNOWN')
            sid = row.get('slide_id', 'UNKNOWN')
            emb = row.get('embedding', 'MISSING')
            
            print(f"  -> Bad Row Index: {idx}")
            print(f"     Patch ID: {pid}")
            print(f"     Class:    {cls}")
            print(f"     Slide ID: {sid}")
            
            # Check the embedding specifically
            if isinstance(emb, float) and np.isnan(emb):
                print(f"     Vector:   NaN (Missing from Cache/Parquet)")
            elif isinstance(emb, np.ndarray):
                # Check inside the vector for infinities
                if not np.all(np.isfinite(emb)):
                    print(f"     Vector:   Corrupt Array (Contains Inf/NaN)")
                    print(f"     First 5:  {emb[:5]}")
            else:
                print(f"     Vector:   {type(emb)} (Unexpected Type)")
                
        raise ValueError("Halting due to corrupted data merge.")

    adata = cohort_to_anndata(cohort_df, ordered_classes)
    
    # 2. Main Point Estimate
    main_result = compute_dpt(adata, root_class, dpt_config)
    
    # 3. Bootstrapping (Conditional)
    boot_samples = {m: [] for m in METRIC_KEYS}
    
    if n_bootstrap > 0:
        # Only log bootstrap progress if it's actually happening (to reduce noise)
        desc = f"    Boot ({embedding_type})"
        for i in range(n_bootstrap):
            boot_ids = dataset.bootstrap_patch_ids(patch_ids, seed=i)
            
            # Reconstruct subset from existing dataframe (Very Fast)
            # We map the bootstrap list to the existing dataframe rows
            # Since get_cohort logic handles list-duplication, we re-call it or do it manually
            boot_df = dataset.get_cohort(boot_ids, model, embedding_type)
            
            boot_adata = cohort_to_anndata(boot_df, ordered_classes)
            result = compute_dpt(boot_adata, root_class, dpt_config)
            
            if result.is_valid:
                for metric in METRIC_KEYS:
                    val = getattr(result, metric)
                    boot_samples[metric].append(val)

    # 4. Construct Output Row
    row = {
        "embedding_type": embedding_type,
        "n_samples": len(patch_ids),
        "bootstrap_iters": n_bootstrap
    }

    for metric in METRIC_KEYS:
        # Point Estimate
        val = getattr(main_result, metric)
        row[metric] = val
        
        # CI (Only populated if n_bootstrap > 0)
        lower, upper = calc_ci(boot_samples[metric])
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper

    return row


def run_full_evaluation() -> pd.DataFrame:
    """
    The Master Loop: Progressions -> Models -> Embeddings.
    """
    
    all_results = []
    
    # --- LEVEL 1: PROGRESSIONS ---
    for prog_config in config.PROGRESSIONS:
        prog_name = prog_config["name"]
        logger.info("=" * 60)
        logger.info(f"Processing Progression: {prog_name}")
        logger.info("=" * 60)
        
        # Initialize Dataset (Fast Mode)
        registry_config = RegistryConfig(
            bucket=prog_config["bucket"],
            prefix=prog_config["prefix"],
            ordered_classes=prog_config["classes"],
            models=config.EXPECTED_MODELS,
            progression_name=prog_name,
            scan_all_models=False
        )
        dataset = ProgressionEmbeddingDataset(registry_config)
        
        # DPT Configuration
        dpt_config = DPTConfig(
            n_neighbors=config.DPT["n_neighbors"],
            n_diffusion_components=config.DPT["n_diffusion_components"],
            metrics=ALL_METRICS,
            subsample_size=2000
        )
        
        # Sample Patches Once per Progression
        patch_ids = dataset.sample_patch_ids(
            n_per_class=config.EVALUATION["n_per_class"],
            max_per_slide=config.EVALUATION["max_per_slide"],
            seed=config.EVALUATION["seed"]
        )
        
        # --- LEVEL 2: MODELS ---
        for model in config.EXPECTED_MODELS:
            logger.info(f"\n  Model: {model}")
            
            try:
                # Load Model into Memory (Single Download)
                dataset.load_model_into_memory(model, patch_ids)
                
                # --- LEVEL 3: EMBEDDING TYPES ---
                pbar = tqdm(ALL_EMBEDDING_TYPES, desc="    Scanning layers", leave=False)
                
                for emb_type in pbar:
                    
                    # Conditional Bootstrapping Logic
                    if emb_type == "final_embedding":
                        n_boot = N_BOOTSTRAPS
                    else:
                        n_boot = 0  # Point estimate only
                    
                    # Run Eval
                    row = evaluate_single_condition(
                        dataset=dataset,
                        patch_ids=patch_ids,
                        model=model,
                        embedding_type=emb_type,
                        ordered_classes=prog_config["classes"],
                        root_class=prog_config["root_class"],
                        dpt_config=dpt_config,
                        n_bootstrap=n_boot
                    )
                    
                    if row:
                        # Add Metadata
                        row["progression"] = prog_name
                        row["model"] = model
                        all_results.append(row)
            
            except Exception as e:
                logger.error(f"Failed to process model {model}: {e}")
            finally:
                # Free memory
                dataset.clear_cache()
                
    return pd.DataFrame(all_results)


def main():
    df = run_full_evaluation()
    
    if not df.empty:
        # Save Raw Results
        output_filename = f"full_manifold_evaluation_{N_BOOTSTRAPS}.csv"
        df.to_csv(output_filename, index=False)
        logger.info(f"\n Evaluation Complete. Saved to {output_filename}")
        
        summary = df[df["embedding_type"] == "final_embedding"][
            ["progression", "model", "tau", "tau_ci_lower", "tau_ci_upper"]
        ].sort_values(["progression", "tau"], ascending=[True, False])
        
        print("\nSummary (Final Embeddings):")
        print(summary.to_markdown(index=False, floatfmt=".3f"))
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()