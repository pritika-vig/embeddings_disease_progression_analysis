#!/usr/bin/env python
"""
Generalizability Test Script: Structural Fidelity vs. Data Efficiency.

Hypothesis:
Models that preserve manifold structure (High Tau) on reference diseases 
will be more data-efficient (Higher F1 with fewer shots) on unseen diseases.

Protocol:
- Leave-One-Out Cross-Validation (LOOCV) across the 4 progressions.
- For each target progression:
    1. Calculate "Reference Score" = Average Tau on the other 3 progressions.
    2. Train Few-Shot Probes (Logistic Regression) on the target progression.
    3. Correlate Reference Tau with Target Few-Shot Performance.
"""

import sys
import logging
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
pcfg.set_icml_style()

OUTPUT_DIR = config.RESULTS_PATH
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = config.FULL_RESULTS_OUTPUT_PATH # Uses your existing full_manifold_evaluation.csv

# Few-Shot Settings
N_SHOTS_LIST = [5, 10, 20] 
N_TRIALS = 10 # Repeat few-shot sampling to reduce noise

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Load & Prep Reference Taus
# -----------------------------------------------------------------------------
def load_reference_taus():
    """
    Loads the main results CSV and pivots it to get a Tau matrix.
    Returns: DataFrame (index=model, columns=progression)
    """
    try:
        df = pd.read_csv(RESULTS_PATH)
        df = df[df["embedding_type"] == "final_embedding"].copy()
    except FileNotFoundError:
        logger.error(f"Could not find results at {RESULTS_PATH}")
        sys.exit(1)
        
    # Pivot to get matrix: Rows=Models, Cols=Progressions, Values=Tau
    tau_matrix = df.pivot(index="model", columns="progression", values="tau")
    
    # Filter only for the 6 core models and 4 core diseases
    tau_matrix = tau_matrix.loc[pcfg.MODEL_ORDER, [p["name"] for p in config.PROGRESSIONS]]
    return tau_matrix

# -----------------------------------------------------------------------------
# 2. Few-Shot Probing Logic
# -----------------------------------------------------------------------------
def train_few_shot_probe(X, y, n_shots, seed):
    """
    Trains a Logistic Regression on n_shots per class.
    """
    # Stratified Split to get exactly n_shots per class (train) vs rest (test)
    # Note: Using StratifiedShuffleSplit is approximate, better to sample manually for exact counts
    
    classes = np.unique(y)
    train_indices = []
    
    rng = np.random.RandomState(seed)
    
    for c in classes:
        c_indices = np.where(y == c)[0]
        if len(c_indices) < n_shots:
            # Fallback if class is tiny (unlikely with this dataset)
            selected = c_indices
        else:
            selected = rng.choice(c_indices, n_shots, replace=False)
        train_indices.extend(selected)
        
    train_idx = np.array(train_indices)
    test_idx = np.setdiff1d(np.arange(len(y)), train_idx)
    
    # Scale Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # Train Linear Probe
    clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=seed)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    return f1_score(y_test, preds, average='macro')

def evaluate_target_performance(target_prog_name, n_shots_list):
    """
    Loads data for ONE target progression and runs few-shot probes for all models.
    """
    prog_config = next(p for p in config.PROGRESSIONS if p["name"] == target_prog_name)
    
    # Initialize Dataset
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=pcfg.MODEL_ORDER,
        progression_name=target_prog_name,
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    
    # Sample IDs once
    patch_ids = dataset.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=42
    )
    
    results = []
    
    for model in pcfg.MODEL_ORDER:
        logger.info(f"  > Probing {model} on {target_prog_name}...")
        try:
            # Load Embeddings
            dataset.load_model_into_memory(model, patch_ids, "final_embedding")
            cohort_df = dataset.get_cohort(patch_ids, model, "final_embedding")
            
            if cohort_df.empty: continue
            
            X = np.vstack(cohort_df["embedding"].values)
            y = cohort_df["class"].values
            
            # Run Trials for each shot count
            for n_shots in n_shots_list:
                scores = []
                for i in range(N_TRIALS):
                    score = train_few_shot_probe(X, y, n_shots, seed=i)
                    scores.append(score)
                
                results.append({
                    "model": model,
                    "n_shots": n_shots,
                    "f1_score": np.mean(scores)
                })
                
        except Exception as e:
            logger.error(f"Failed {model}: {e}")
        finally:
            dataset.clear_cache()
            
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    # 1. Load Reference Taus
    tau_matrix = load_reference_taus()
    logger.info("Loaded Reference Taus.")
    
    all_results = []
    
    # 2. Leave-One-Out Loop
    progressions = [p["name"] for p in config.PROGRESSIONS]
    
    for target in progressions:
        logger.info(f"\n=== Leave-One-Out Target: {target} ===")
        
        # A. Calculate Reference Score (Avg Tau of OTHERS)
        others = [p for p in progressions if p != target]
        ref_scores = tau_matrix[others].mean(axis=1).reset_index(name="ref_tau_avg")
        
        # B. Measure Target Performance
        target_perf_df = evaluate_target_performance(target, N_SHOTS_LIST)
        
        if target_perf_df.empty:
            continue
            
        # C. Merge
        merged = pd.merge(target_perf_df, ref_scores, on="model")
        merged["target_progression"] = target
        all_results.append(merged)
        
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_DIR / "generalizability_results.csv", index=False)
    else:
        logger.error("No results generated.")

if __name__ == "__main__":
    main()