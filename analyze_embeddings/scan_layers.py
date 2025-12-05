#!/usr/bin/env python
"""
Layer-wise Diffusion Pseudotime Scan.

Generates evolution plots of Kendall's Tau across network depths (0.125 -> 1.0)
for Patch, CLS, and Register tokens across all models.
"""

import logging
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import config
from analysis.dpt import (
    DPTConfig,
    DPTMetric,
    cohort_to_anndata,
    compute_dpt,
)
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
DEPTHS = [0.125, 0.375, 0.625, 0.875, 1.0]

# Maps readable names to the f-string format used in your embedding dictionary
TOKEN_TYPES = {
    "Patch Mean": "patch_mean_{}",
    "CLS Token": "cls_{}",
    "Register Mean": "register_mean_{}",
}


def get_progression_config(name: str) -> dict:
    for prog in config.PROGRESSIONS:
        if prog["name"] == name:
            return prog
    raise ValueError(f"Progression '{name}' not found in config.PROGRESSIONS")


def run_layer_scan(progression_name: str) -> pd.DataFrame:
    """
    Compute Kendall's Tau for all models, token types, and depths.
    Uses point estimates (no bootstrapping) for speed.
    """
    logger.info("=" * 60)
    logger.info(f"  Scanning Layers: {progression_name}")
    logger.info("=" * 60)

    # 1. Setup Dataset & Config
    prog = get_progression_config(progression_name)
    registry_config = RegistryConfig(
        bucket=prog["bucket"],
        prefix=prog["prefix"],
        ordered_classes=prog["classes"],
        models=config.EXPECTED_MODELS,
        progression_name=progression_name,
    )
    dataset = ProgressionEmbeddingDataset(registry_config)

    # Minimal DPT Config: Only compute TAU for speed
    dpt_config = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=config.DPT["n_diffusion_components"],
        metrics={DPTMetric.TAU},  # <--- ONLY TAU
        subsample_size=2000 # Unused since we skip expensive metrics
    )

    # 2. Sample Patches ONCE
    # Critical: Use the exact same set of patch IDs for every layer/model 
    # to ensure the comparison is rigorous.
    logger.info("Sampling consistent patch set...")
    patch_ids = dataset.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=config.EVALUATION["seed"],
    )

    results = []
    
    # 3. Execution Loop
    # Iterate: Model -> Token -> Depth
    total_steps = len(config.EXPECTED_MODELS) * len(TOKEN_TYPES) * len(DEPTHS)
    pbar = tqdm(total=total_steps, desc="Processing Layers")

    for model in config.EXPECTED_MODELS:
        dataset.load_model_into_memory(model, patch_ids)
        for token_label, token_fmt in TOKEN_TYPES.items():
            for depth in DEPTHS:
                embedding_key = token_fmt.format(depth)
                
                tau = float("nan")
                try:
                    # Fetch data for this specific layer
                    cohort_df = dataset.get_cohort(
                        patch_ids, 
                        model=model, 
                        embedding_type=embedding_key
                    )
                    
                    # Convert to AnnData
                    adata = cohort_to_anndata(cohort_df, prog["classes"])
                    
                    # Compute DPT
                    result = compute_dpt(adata, prog["root_class"], dpt_config)
                    
                    if result.is_valid:
                        tau = result.tau
                        
                except Exception as e:
                    # Some models (like ResNet) might not have 'register' tokens.
                    # We catch this silently to allow the scan to proceed.
                    # logger.debug(f"Skipping {model} {embedding_key}: {e}")
                    pass
                
                results.append({
                    "Model": model,
                    "Token Type": token_label,
                    "Depth": depth,
                    "Tau": tau
                })
                pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)


def plot_layer_evolution(df: pd.DataFrame, progression_name: str):
    """
    Generate a 3-panel plot (one per token type) showing Tau over Depth.
    """
    if df.empty or df["Tau"].isnull().all():
        logger.warning("No valid data to plot.")
        return

    # Setup Plot Style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Create FacetGrid: Columns = Token Type
    g = sns.relplot(
        data=df,
        x="Depth",
        y="Tau",
        hue="Model",
        col="Token Type",
        kind="line",
        marker="o",
        dashes=False,
        linewidth=2.5,
        palette="tab10",
        height=5,
        aspect=1.0,
        facet_kws={"sharey": True, "sharex": True}
    )

    # Customizations
    g.fig.suptitle(f"Manifold Quality Evolution: {progression_name}", y=1.05, fontsize=16)
    g.set_axis_labels("Network Depth (Normalized)", "Kendall's Tau")
    g.set_titles("{col_name}")
    
    # Add reference line at Tau=0 (Random)
    for ax in g.axes.flat:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Save
    filename = f"layer_evolution_{progression_name}.png"
    g.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"\nGraph saved to: {filename}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_layers.py <progression_name>")
        print(f"Available: {[p['name'] for p in config.PROGRESSIONS]}")
        sys.exit(1)
    
    prog_name = sys.argv[1]
    
    # Run Scan
    df = run_layer_scan(prog_name)
    
    # Save Raw Data
    csv_path = f"layer_scan_{prog_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw data saved to: {csv_path}")
    
    # Generate Plots
    plot_layer_evolution(df, prog_name)


if __name__ == "__main__":
    main()