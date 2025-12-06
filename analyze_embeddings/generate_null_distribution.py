#!/usr/bin/env python
"""
Null Distribution Generator.

Methodology:
1. Load Embeddings & Construct Manifold (DPT) on REAL data.
2. Fix the calculated Pseudotime coordinates.
3. Permute (shuffle) the Class Labels N times.
4. Calculate Kendall's Tau for each permutation.
"""

import logging
import sys
import shutil
from pathlib import Path
from typing import Optional

# --- IMPORT FIX ---
# Ensure we can see the parent directory where config.py lives
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm

try:
    import config
except ImportError:
    print("❌ Critical Error: Could not import 'config.py'. Run this from the project root or ensure python path is set.")
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

# Configuration
N_BOOTSTRAP_NULL = 10  
EMBEDDING_TARGET = "final_embedding"

def get_output_path() -> Path:
    """Safely resolve the output path from config, validating early."""
    try:
        # Check if attribute exists
        if not hasattr(config, "NULL_RESULTS_OUTPUT_PATH"):
            # Fallback if config isn't updated in the running process
            default_path = config.RESULTS_DIR / "null_manifold_evaluation.csv"
            logger.warning(f"⚠️ 'NULL_RESULTS_OUTPUT_PATH' missing in config. Using fallback: {default_path}")
            return default_path
            
        return config.NULL_RESULTS_OUTPUT_PATH
    except Exception as e:
        logger.error(f"❌ Configuration Error: {e}")
        sys.exit(1)

def run_null_evaluation():
    # --- STEP 0: VALIDATE PATHS BEFORE WORK STARTS ---
    output_path = get_output_path()
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check write permissions by touching a temp file
    try:
        test_file = output_path.parent / ".perm_check"
        test_file.touch()
        test_file.unlink()
        logger.info(f"✅ Output path validated: {output_path}")
    except PermissionError:
        logger.error(f"❌ Cannot write to {output_path.parent}. Check permissions.")
        sys.exit(1)

    all_null_results = []
    
    # --- LEVEL 1: PROGRESSIONS ---
    for prog_config in config.PROGRESSIONS:
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
            n_per_class=config.EVALUATION["n_per_class"],
            max_per_slide=config.EVALUATION["max_per_slide"],
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
                
                for _ in range(N_BOOTSTRAP_NULL):
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
                    "n_permutations": N_BOOTSTRAP_NULL
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
        logger.info(f"\n✅ Null Evaluation Complete. Saved to {output_path}")
    else:
        logger.warning("No null results generated.")

if __name__ == "__main__":
    run_null_evaluation()