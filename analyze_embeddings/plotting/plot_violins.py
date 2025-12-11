#!/usr/bin/env python
"""
Plotting Script: Pseudotime Distribution Violins.
Visualizes the "Semantic Collapse" effect.

Target:
- Progression: CRC-Serrated
- Layer: final_embedding
- Standards: Uses plotting/plot_config.py for colors, order, and styling.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import (
    DPTConfig,
    cohort_to_anndata,
    compute_dpt,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Apply ICML Styling immediately
pcfg.set_icml_style()

TARGET_PROGRESSION = "CRC-Serrated" 
TARGET_EMBEDDING = "final_embedding"
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def collect_raw_pseudotimes():
    # 1. Setup Dataset
    # Find the config dict for this progression
    prog_config = next(p for p in config.PROGRESSIONS if p["name"] == TARGET_PROGRESSION)
    biological_order = prog_config["classes"] # Capture the correct biological order
    
    logger.info(f"Loading Data for {TARGET_PROGRESSION}...")
    
    # We scan all models defined in the plot config to ensure we have data for the plot
    models_to_scan = pcfg.MODEL_ORDER
    
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=biological_order,
        models=models_to_scan,
        progression_name=TARGET_PROGRESSION,
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    
    patch_ids = dataset.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=config.EVALUATION["seed"]
    )
    
    dpt_config = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=config.DPT["n_diffusion_components"],
        metrics={}, 
        subsample_size=2000
    )

    all_data = []

    # 2. Iterate Models (Using Plot Config Order)
    for model in models_to_scan:
        logger.info(f"Processing Model: {model}")
        
        try:
            # Load embeddings
            dataset.load_model_into_memory(model, patch_ids, TARGET_EMBEDDING)
            cohort_df = dataset.get_cohort(patch_ids, model, TARGET_EMBEDDING)
            
            if cohort_df.empty:
                continue

            # Run DPT
            # Important: cohort_to_anndata uses biological_order to set categories
            adata = cohort_to_anndata(cohort_df, biological_order)
            compute_dpt(adata, prog_config["root_class"], dpt_config)
            
            # 3. Extract Raw Data
            if "dpt_pseudotime" in adata.obs:
                extract_df = pd.DataFrame({
                    "dpt_pseudotime": adata.obs["dpt_pseudotime"].values,
                    "stage": adata.obs["class"].values, 
                    "stage_int": adata.obs["stage_int"].values,
                    "model": model
                })
                
                # Filter out inf/nans
                extract_df = extract_df[np.isfinite(extract_df["dpt_pseudotime"])]
                
                # Normalize Pseudotime to 0-1
                pt = extract_df["dpt_pseudotime"]
                if pt.max() > pt.min():
                    extract_df["dpt_pseudotime"] = (pt - pt.min()) / (pt.max() - pt.min())
                else:
                    extract_df["dpt_pseudotime"] = 0.0
                
                all_data.append(extract_df)
                
        except Exception as e:
            logger.error(f"Error processing {model}: {e}")
        finally:
            dataset.clear_cache()
            
    return pd.concat(all_data, ignore_index=True), biological_order

def plot_violins(df, order):
    """Generates the Violin Grid using pcfg standards."""
    logger.info("Generating Plots...")
    
    # 1. Filter & Sort by Config Order
    df = df[df['model'].isin(pcfg.MODEL_ORDER)].copy()
    df['model'] = pd.Categorical(df['model'], categories=pcfg.MODEL_ORDER, ordered=True)
    
    # 2. Setup Grid
    # We have 6 models in pcfg.MODEL_ORDER, so 2 rows x 3 cols is perfect
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    # 3. Plot Loop
    for i, model in enumerate(pcfg.MODEL_ORDER):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        
        subset = df[df['model'] == model]
        if subset.empty:
            continue
            
        # Get specific color from config
        color = pcfg.MODEL_COLORS.get(model, '#333333')
        
        # VIOLIN PLOT
        # 'order' ensures the X-axis respects biological progression
        sns.violinplot(
            data=subset,
            x='stage',
            y='dpt_pseudotime',
            order=order,      # <--- Enforces Biological Order
            ax=ax,
            inner="quartile",
            color=color,      # <--- Uses Specific Model Color
            alpha=0.7,
            linewidth=1.2,
            cut=0 
        )
        
        # Aesthetics
        label = pcfg.MODEL_LABELS.get(model, model)
        ax.set_title(label, fontweight='bold', fontsize=14, pad=10)
        
        # Y-Label only on left columns
        if i % 3 == 0:
            ax.set_ylabel("Diffusion Pseudotime (0-1)", fontweight='bold')
        else:
            ax.set_ylabel("")
            
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        sns.despine(ax=ax)
        
        # X-Labels only on bottom row, rotated for readability
        if i >= 3:
            ax.set_xticklabels(order, rotation=20, ha='right')
        else:
            ax.set_xlabel("")

    # Global Title
    plt.suptitle(f"Pseudotime Distribution: {TARGET_PROGRESSION}\n(Vision Models vs. Vision-Language Collapse)", 
                 fontsize=16, fontweight='bold', y=0.96)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    save_path = OUTPUT_DIR / f'violin_pseudotime_{TARGET_PROGRESSION}_styled.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved to {save_path}")

if __name__ == "__main__":
    df_results, biological_order = collect_raw_pseudotimes()
    if not df_results.empty:
        plot_violins(df_results, biological_order)
    else:
        logger.error("No data collected.")