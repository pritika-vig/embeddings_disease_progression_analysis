#!/usr/bin/env python
"""
Hyperparameter Sweep Script: DPT Neighbors (k)
Generates CSV results and Visualization Plots.

Usage:
    python generate_data/hyper_param_sweep.py
    python generate_data/hyper_param_sweep.py --progression CRC-Serrated
    python generate_data/hyper_param_sweep.py --k_values 10 50 100 200
    python generate_data/hyper_param_sweep.py --output custom_path.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set, Dict, Any

# --- Import Path Setup ---
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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

TARGET_PROGRESSION = "CRC-Serrated"
TARGET_EMBEDDING = "final_embedding"
K_VALUES = [10, 30, 50, 75, 100, 200]

METRIC_KEYS = ["tau", "trustworthiness", "spectral_gap"]
ALL_METRICS = {
    DPTMetric.TAU,
    DPTMetric.TRUSTWORTHINESS,
    DPTMetric.SPECTRAL_GAP
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def find_progression_config(name: str):
    for p in config.PROGRESSIONS:
        if p["name"] == name:
            return p
    raise ValueError(f"Progression '{name}' not found in config.")

def evaluate_k(
    cohort_df: pd.DataFrame,
    ordered_classes: List[str],
    root_class: str,
    k: int,
) -> Dict[str, float]:
    """Runs DPT for a specific value of K."""
    adata = cohort_to_anndata(cohort_df, ordered_classes)

    dpt_config = DPTConfig(
        n_neighbors=k,
        n_diffusion_components=10,
        metrics=ALL_METRICS,
        subsample_size=1000
    )

    result = compute_dpt(adata, root_class, dpt_config)

    output = {}
    if result.is_valid:
        for m in METRIC_KEYS:
            output[m] = getattr(result, m)
    else:
        for m in METRIC_KEYS:
            output[m] = np.nan
    return output

def plot_sweep_results(df: pd.DataFrame, output_path: str = "k_sweep_plot.png"):
    """Generates publication-quality plots from the sweep dataframe."""
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Trajectory Fidelity (Tau)
    sns.lineplot(
        data=df,
        x="k",
        y="tau",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        ax=axes[0],
        linewidth=2.5,
        palette="viridis"
    )
    axes[0].set_title("Trajectory Fidelity (Kendall's tau) vs. Neighbors (k)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Number of Neighbors (k)", fontsize=12)
    axes[0].set_ylabel("Kendall's tau", fontsize=12)
    axes[0].legend(title="Model", loc="best", fontsize=10)

    # Plot 2: Trustworthiness
    sns.lineplot(
        data=df,
        x="k",
        y="trustworthiness",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        ax=axes[1],
        linewidth=2.5,
        palette="viridis"
    )
    axes[1].set_title("Manifold Trustworthiness vs. Neighbors (k)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Number of Neighbors (k)", fontsize=12)
    axes[1].set_ylabel("Trustworthiness", fontsize=12)
    axes[1].get_legend().remove() # Remove legend from second plot to save space

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Plots saved to {output_path}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def run_sweep(progression: str = TARGET_PROGRESSION, k_values: List[int] = None) -> pd.DataFrame:
    if k_values is None:
        k_values = K_VALUES

    all_results = []

    prog_config = find_progression_config(progression)
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=config.EXPECTED_MODELS,
        progression_name=progression,
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)

    # Use fixed seed for consistent patch selection across K values
    patch_ids = dataset.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=config.EVALUATION["seed"]
    )

    for model in config.EXPECTED_MODELS:
        logger.info(f"\nModel: {model}")
        try:
            dataset.load_model_into_memory(model, patch_ids, TARGET_EMBEDDING)
            cohort_df = dataset.get_cohort(patch_ids, model=model, embedding_type=TARGET_EMBEDDING)

            if cohort_df.empty: continue

            pbar = tqdm(k_values, desc="  Sweeping K", leave=False)
            for k in pbar:
                if k >= len(cohort_df): continue

                metrics = evaluate_k(cohort_df, prog_config["classes"], prog_config["root_class"], k)
                row = {"model": model, "k": k, "n_samples": len(cohort_df)}
                row.update(metrics)
                all_results.append(row)

        except Exception as e:
            logger.error(f"Failed model {model}: {e}")
        finally:
            dataset.clear_cache()

    return pd.DataFrame(all_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DPT hyperparameter sweep over k (n_neighbors)."
    )
    parser.add_argument(
        "--progression", type=str, default=TARGET_PROGRESSION,
        help=f"Target progression (default: {TARGET_PROGRESSION})"
    )
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=K_VALUES,
        help=f"k values to sweep (default: {K_VALUES})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output CSV path"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = run_sweep(progression=args.progression, k_values=args.k_values)

    if not df.empty:
        if args.output:
            csv_path = Path(args.output)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path = csv_path.parent / "k_sweep_analysis.png"
        else:
            output_dir = config.get_output_dir("hyperparam_sweep")
            csv_path = output_dir / "hyperparam_sweep_k_results.csv"
            plot_path = output_dir / "k_sweep_analysis.png"

        # Save Data
        df.to_csv(csv_path, index=False)
        logger.info(f"Sweep data saved to {csv_path}")

        # Generate Plot
        plot_sweep_results(df, output_path=str(plot_path))

        # Preview
        print("\nResults Preview:")
        print(df.head(10).to_markdown(index=False, floatfmt=".3f"))
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()
