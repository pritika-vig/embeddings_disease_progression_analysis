#!/usr/bin/env python
"""
Trajectory Comparison Inspector.

Compares two models on the same progression, showing how each orders
patches along the inferred trajectory. Useful for visualizing the
difference between high-tau and low-tau models.

Usage:
    python compare_trajectories.py
    python compare_trajectories.py --model1 virchow2 --model2 conch --progression BDC
    python compare_trajectories.py --sampling stratified --seed 42
"""

import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import time

# Plotting & Image Imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import fsspec

# Ensure we can import config from parent directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("❌ Critical Error: Could not import 'config.py'.")
    sys.exit(1)

from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import (
    DPTConfig,
    DPTMetric,
    DPTResult,
    cohort_to_anndata,
    compute_dpt,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Edit these defaults as needed
# =============================================================================

# Default models to compare
MODEL_HIGH_TAU = "uni2"      # Typically high trajectory fidelity
MODEL_LOW_TAU = "dinov2"     # Typically lower trajectory fidelity

# Default progression to analyze
TARGET_PROGRESSION = "SCC"   # Options: "SCC", "CRC-Conventional", "CRC-Serrated", "BDC"

# Number of patches to sample along trajectory
NUM_SAMPLES = 5

# Sampling strategy: "uniform", "quantile", "stratified", "random"
DEFAULT_SAMPLING = "quantile"

# =============================================================================

EMBEDDING_TYPE = "final_embedding"

# Stage colors for visual distinction
STAGE_COLORS = {
    0: '#2ecc71',  # Green - early/normal
    1: '#f39c12',  # Orange - intermediate
    2: '#e74c3c',  # Red - late/invasive
    3: '#9b59b6',  # Purple - if 4 stages
}

# Sampling strategy descriptions
SAMPLING_STRATEGIES = {
    "uniform": "Evenly spaced in pseudotime value (0.0, 0.25, 0.5, ...)",
    "quantile": "Evenly spaced in rank (adapts to data distribution)",
    "stratified": "At least one patch per stage, rest fill gaps",
    "random": "Random selection across the trajectory",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DPT trajectories for two models on the same progression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sampling strategies:
  uniform    - Evenly spaced in pseudotime value (may miss sparse stages)
  quantile   - Evenly spaced in rank/percentile (adapts to data distribution)  
  stratified - Guarantees at least one patch per stage
  random     - Random selection (use --seed for reproducibility)
        """
    )
    parser.add_argument(
        "--model1", 
        type=str, 
        default=MODEL_HIGH_TAU,
        choices=config.EXPECTED_MODELS,
        help=f"First model, typically high-tau (default: {MODEL_HIGH_TAU})"
    )
    parser.add_argument(
        "--model2", 
        type=str, 
        default=MODEL_LOW_TAU,
        choices=config.EXPECTED_MODELS,
        help=f"Second model, typically low-tau (default: {MODEL_LOW_TAU})"
    )
    parser.add_argument(
        "--progression", 
        type=str, 
        default=TARGET_PROGRESSION,
        help=f"Name of the progression (default: {TARGET_PROGRESSION})"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=NUM_SAMPLES,
        help=f"Number of patches to sample along each trajectory (default: {NUM_SAMPLES})"
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default=DEFAULT_SAMPLING,
        choices=list(SAMPLING_STRATEGIES.keys()),
        help=f"Sampling strategy (default: {DEFAULT_SAMPLING})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None = different each run)"
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.05,
        help="For uniform/quantile: randomness range around target point (default: 0.05)"
    )
    return parser.parse_args()


def sample_patches_uniform(trajectory_df: pd.DataFrame, num_samples: int, 
                           rng: np.random.Generator, jitter: float = 0.05) -> pd.DataFrame:
    """
    Sample patches at evenly-spaced pseudotime VALUES with optional jitter.
    
    Args:
        trajectory_df: DataFrame sorted by dpt_pseudotime_norm
        num_samples: Number of patches to select
        rng: Numpy random generator
        jitter: Random range around each target point (0 = exact, 0.1 = ±10%)
    """
    pseudotimes = trajectory_df["dpt_pseudotime_norm"].values
    target_values = np.linspace(0, 1, num_samples)
    
    selected_ilocs = []
    for target in target_values:
        # Add jitter: random offset within ±jitter range
        if jitter > 0:
            offset = rng.uniform(-jitter, jitter)
            target_jittered = np.clip(target + offset, 0, 1)
        else:
            target_jittered = target
        
        # Find closest patch
        idx = (np.abs(pseudotimes - target_jittered)).argmin()
        selected_ilocs.append(idx)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ilocs = [x for x in selected_ilocs if not (x in seen or seen.add(x))]
    
    return trajectory_df.iloc[unique_ilocs].copy()


def sample_patches_quantile(trajectory_df: pd.DataFrame, num_samples: int,
                            rng: np.random.Generator, jitter: float = 0.05) -> pd.DataFrame:
    """
    Sample patches at evenly-spaced RANKS (percentiles) with optional jitter.
    
    This adapts to the actual distribution of data - if most patches cluster
    in early pseudotime, we'll still sample across all stages proportionally.
    """
    n_total = len(trajectory_df)
    target_ranks = np.linspace(0, n_total - 1, num_samples).astype(int)
    
    selected_ilocs = []
    for rank in target_ranks:
        # Add jitter in rank space
        if jitter > 0:
            max_offset = int(n_total * jitter)
            offset = rng.integers(-max_offset, max_offset + 1)
            rank_jittered = np.clip(rank + offset, 0, n_total - 1)
        else:
            rank_jittered = rank
        
        selected_ilocs.append(rank_jittered)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ilocs = [x for x in selected_ilocs if not (x in seen or seen.add(x))]
    
    return trajectory_df.iloc[unique_ilocs].copy()


def sample_patches_stratified(trajectory_df: pd.DataFrame, num_samples: int,
                              rng: np.random.Generator) -> pd.DataFrame:
    """
    Sample patches ensuring at least one from each stage.
    
    Strategy:
    1. Randomly select one patch per stage
    2. Fill remaining slots with random patches across trajectory
    3. Sort final selection by pseudotime
    """
    stages = trajectory_df["stage_int"].unique()
    n_stages = len(stages)
    
    if num_samples < n_stages:
        logger.warning(f"num_samples ({num_samples}) < n_stages ({n_stages}). "
                      f"Increasing to {n_stages}.")
        num_samples = n_stages
    
    selected_indices = []
    
    # Step 1: One random patch per stage
    for stage in sorted(stages):
        stage_df = trajectory_df[trajectory_df["stage_int"] == stage]
        idx = rng.choice(stage_df.index)
        selected_indices.append(idx)
    
    # Step 2: Fill remaining slots randomly
    remaining = num_samples - len(selected_indices)
    if remaining > 0:
        available = trajectory_df.index.difference(selected_indices)
        if len(available) >= remaining:
            extra = rng.choice(available, size=remaining, replace=False)
            selected_indices.extend(extra)
        else:
            selected_indices.extend(available)
    
    # Step 3: Sort by pseudotime for display
    selected_df = trajectory_df.loc[selected_indices].copy()
    selected_df = selected_df.sort_values("dpt_pseudotime_norm")
    
    return selected_df


def sample_patches_random(trajectory_df: pd.DataFrame, num_samples: int,
                          rng: np.random.Generator) -> pd.DataFrame:
    """
    Randomly sample patches and sort by pseudotime.
    """
    n_total = len(trajectory_df)
    num_samples = min(num_samples, n_total)
    
    selected_ilocs = rng.choice(n_total, size=num_samples, replace=False)
    selected_df = trajectory_df.iloc[sorted(selected_ilocs)].copy()
    
    return selected_df


def sample_patches(trajectory_df: pd.DataFrame, num_samples: int, 
                   strategy: str, rng: np.random.Generator, jitter: float = 0.05) -> pd.DataFrame:
    """
    Sample patches using the specified strategy.
    """
    if strategy == "uniform":
        return sample_patches_uniform(trajectory_df, num_samples, rng, jitter)
    elif strategy == "quantile":
        return sample_patches_quantile(trajectory_df, num_samples, rng, jitter)
    elif strategy == "stratified":
        return sample_patches_stratified(trajectory_df, num_samples, rng)
    elif strategy == "random":
        return sample_patches_random(trajectory_df, num_samples, rng)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def compute_trajectory_for_model(
    prog_config: dict,
    model_name: str,
    patch_ids: list,
    num_samples: int,
    sampling_strategy: str,
    rng: np.random.Generator,
    jitter: float
) -> tuple[pd.DataFrame, float]:
    """
    Compute DPT trajectory for a single model and return selected patches + tau.
    
    Returns:
        Tuple of (selected_patches_df, tau_value)
    """
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=[model_name],
        progression_name=prog_config["name"],
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    
    tau_value = np.nan
    
    try:
        dataset.load_model_into_memory(model_name, patch_ids, EMBEDDING_TYPE)
        cohort_df = dataset.get_cohort(patch_ids, model_name, EMBEDDING_TYPE)
        
        if cohort_df.empty:
            logger.error(f"No data found for {model_name}")
            return pd.DataFrame(), tau_value
        
        adata = cohort_to_anndata(cohort_df, prog_config["classes"])
        
        # Configure DPT with TAU metric enabled
        dpt_conf = DPTConfig(
            n_neighbors=config.DPT["n_neighbors"],
            n_diffusion_components=config.DPT["n_diffusion_components"],
            metrics={DPTMetric.TAU},
            subsample_size=2000
        )
        
        # Compute DPT and get result with tau
        dpt_result = compute_dpt(adata, prog_config["root_class"], dpt_conf)
        tau_value = dpt_result.tau
        
        logger.info(f"   {model_name}: τ = {tau_value:.3f}")
        
        if "dpt_pseudotime" not in adata.obs:
            logger.error(f"DPT failed for {model_name}")
            return pd.DataFrame(), tau_value
        
        # Extract and normalize pseudotime to [0, 1]
        trajectory_df = adata.obs.copy()
        trajectory_df = trajectory_df[np.isfinite(trajectory_df["dpt_pseudotime"])]
        
        pt_min = trajectory_df["dpt_pseudotime"].min()
        pt_max = trajectory_df["dpt_pseudotime"].max()
        trajectory_df["dpt_pseudotime_norm"] = (
            (trajectory_df["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)
        )
        
        trajectory_df = trajectory_df.sort_values("dpt_pseudotime_norm")
        
        # Sample patches using selected strategy
        selected_patches = sample_patches(
            trajectory_df, num_samples, sampling_strategy, rng, jitter
        )
        selected_patches['model'] = model_name
        
        return selected_patches, tau_value
        
    finally:
        dataset.clear_cache()


def fetch_image(fs, gcs_path: str) -> Image.Image:
    """Fetch image from GCS, return None if failed."""
    try:
        with fs.open(gcs_path, 'rb') as f:
            return Image.open(f).convert("RGB")
    except Exception as e:
        logger.warning(f"Could not load: {gcs_path}: {e}")
        return None


def plot_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    model1: str,
    model2: str,
    tau1: float,
    tau2: float,
    prog_config: dict,
    output_path: Path,
    sampling_strategy: str,
    seed: int
):
    """
    Create side-by-side comparison figure with two rows of patches.
    """
    fs = fsspec.filesystem("gcs")
    bucket = prog_config["bucket"]
    image_subdir = prog_config.get("image_subdir", "imagenet")
    
    n1, n2 = len(df1), len(df2)
    n_cols = max(n1, n2)
    
    # Figure setup: 2 rows (one per model) + space for labels
    fig_width = max(12, n_cols * 2.2)
    fig_height = 6.5
    
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, fig_height))
    
    # Handle single column edge case
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    # Title for the whole figure
    progression_name = prog_config["name"]
    seed_str = f"seed={seed}" if seed is not None else "random"
    fig.suptitle(
        f"Trajectory Comparison: {progression_name}\n"
        f"(sampling: {sampling_strategy}, {seed_str})",
        fontsize=13,
        fontweight='bold',
        y=0.99
    )
    
    def plot_row(row_idx, df, model_name, tau_val):
        """Plot one row of patches."""
        if np.isfinite(tau_val):
            tau_str = f"(τ = {tau_val:.2f})"
        else:
            tau_str = "(τ = N/A)"
        row_label = f"{model_name}\n{tau_str}"
        
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx >= len(df):
                ax.axis('off')
                continue
            
            row = df.iloc[col_idx]
            
            # Build GCS path
            cls = row['class']
            slide_id = str(row['slide_id'])
            patch_id = str(row['patch_id'])
            
            if not slide_id.startswith("slide_"):
                slide_id = f"slide_{slide_id}"
            if not patch_id.startswith("patch_"):
                patch_id = f"patch_{patch_id}"
            
            gcs_path = f"gs://{bucket}/{image_subdir}/train/{cls}/{slide_id}/{patch_id}.png"
            
            # Fetch and display image
            img = fetch_image(fs, gcs_path)
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Not\nFound", ha='center', va='center', 
                       fontsize=10, color='red')
                ax.set_facecolor('#f0f0f0')
            
            # Get stage info
            stage_val = row.get('stage_int', -1)
            stage = int(stage_val) if pd.notna(stage_val) else -1
            pt_norm = row.get('dpt_pseudotime_norm', row.get('dpt_pseudotime', 0))
            
            # Color border by stage
            border_color = STAGE_COLORS.get(stage, '#95a5a6')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(3)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Labels below image
            if row_idx == 1:  # Bottom row gets pseudotime labels
                ax.set_xlabel(f"PT: {pt_norm:.2f}", fontsize=9)
            
            # Stage label above image
            stage_label = f"Stage {stage}" if stage >= 0 else "?"
            ax.set_title(stage_label, fontsize=10, color=border_color, fontweight='bold')
        
        # Row label on the left
        axes[row_idx, 0].set_ylabel(
            row_label, 
            fontsize=11, 
            fontweight='bold',
            rotation=0,
            ha='right',
            va='center',
            labelpad=60
        )
    
    # Plot both rows
    plot_row(0, df1, model1, tau1)
    plot_row(1, df2, model2, tau2)
    
    # Add legend for stages
    stage_names = prog_config.get("classes", [])
    legend_elements = []
    for i, name in enumerate(stage_names):
        color = STAGE_COLORS.get(i, '#95a5a6')
        # Truncate long class names
        display_name = (name[:25] + '...') if len(name) > 28 else name
        legend_elements.append(
            plt.Line2D([0], [0], marker='s', color='w', 
                      markerfacecolor=color, markersize=10,
                      label=f"Stage {i}: {display_name}")
        )
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=min(len(stage_names), 4),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02)
    )
    
    plt.tight_layout(rect=[0.08, 0.08, 1, 0.93])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Comparison plot saved to {output_path}")


def main():
    args = parse_args()
    
    model1 = args.model1
    model2 = args.model2
    target_progression = args.progression
    num_samples = args.num_samples
    sampling_strategy = args.sampling
    jitter = args.jitter
    
    # Set up random seed
    if args.seed is not None:
        seed = args.seed
        logger.info(f"🎲 Using fixed seed: {seed}")
    else:
        seed = int(time.time() * 1000) % (2**32)
        logger.info(f"🎲 Using random seed: {seed} (pass --seed {seed} to reproduce)")
    
    rng = np.random.default_rng(seed)
    
    # Find progression config
    prog_config = next(
        (p for p in config.PROGRESSIONS if p["name"] == target_progression),
        None
    )
    
    if not prog_config:
        available = [p["name"] for p in config.PROGRESSIONS]
        logger.error(f"❌ Progression '{target_progression}' not found.")
        logger.error(f"   Available: {available}")
        sys.exit(1)
    
    logger.info(f"🔍 Comparing Trajectories on {target_progression}")
    logger.info(f"   Model 1 (high-tau expected): {model1}")
    logger.info(f"   Model 2 (low-tau expected):  {model2}")
    logger.info(f"   Sampling strategy: {sampling_strategy}")
    logger.info(f"   {SAMPLING_STRATEGIES[sampling_strategy]}")
    
    # Get shared patch IDs (same patches for both models)
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=[model1],
        progression_name=target_progression,
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    
    patch_ids = dataset.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"],
        max_per_slide=config.EVALUATION["max_per_slide"],
        seed=config.EVALUATION["seed"]
    )
    dataset.clear_cache()
    
    logger.info(f"   Using {len(patch_ids)} shared patches\n")
    
    # Compute trajectories for both models
    logger.info(f"⚙️  Computing trajectory for {model1}...")
    df1, tau1 = compute_trajectory_for_model(
        prog_config, model1, patch_ids, num_samples, 
        sampling_strategy, rng, jitter
    )
    
    logger.info(f"⚙️  Computing trajectory for {model2}...")
    df2, tau2 = compute_trajectory_for_model(
        prog_config, model2, patch_ids, num_samples,
        sampling_strategy, rng, jitter
    )
    
    if df1.empty or df2.empty:
        logger.error("❌ Failed to compute one or both trajectories.")
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"TRAJECTORY COMPARISON: {target_progression}")
    print(f"Sampling: {sampling_strategy} | Seed: {seed}")
    print("=" * 80)
    print(f"\n{'Model':<15} | {'Kendall τ':<12} | {'Interpretation'}")
    print("-" * 50)
    print(f"{model1:<15} | {tau1:<12.3f} | {'Higher fidelity' if tau1 > tau2 else 'Lower fidelity'}")
    print(f"{model2:<15} | {tau2:<12.3f} | {'Higher fidelity' if tau2 > tau1 else 'Lower fidelity'}")
    
    for model_name, df, tau in [(model1, df1, tau1), (model2, df2, tau2)]:
        print(f"\n{model_name} (τ = {tau:.3f}):")
        print(f"  {'PT (norm)':<10} | {'Stage':<6} | {'Class'}")
        print(f"  {'-'*50}")
        for _, row in df.iterrows():
            pt = row.get('dpt_pseudotime_norm', row.get('dpt_pseudotime', 0))
            stage = int(row['stage_int']) if pd.notna(row.get('stage_int')) else '?'
            cls = row['class'][:40]
            print(f"  {pt:<10.3f} | {stage:<6} | {cls}")
    
    print("\n" + "=" * 80 + "\n")
    
    # Generate comparison plot
    seed_suffix = f"_seed{seed}" if args.seed is not None else ""
    plot_filename = f"comparison_{model1}_vs_{model2}_{target_progression.replace(' ', '_')}_{sampling_strategy}{seed_suffix}.png"
    plot_path = config.RESULTS_DIR / "plots" / plot_filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(
        df1, df2,
        model1, model2,
        tau1, tau2,
        prog_config,
        plot_path,
        sampling_strategy,
        seed if args.seed is not None else None
    )
    
    # Print conclusion
    if tau1 > tau2:
        better, worse = model1, model2
        better_tau, worse_tau = tau1, tau2
    else:
        better, worse = model2, model1
        better_tau, worse_tau = tau2, tau1
    
    print(f"📊 Summary:")
    print(f"   {better} (τ={better_tau:.2f}) better preserves disease progression order")
    print(f"   {worse} (τ={worse_tau:.2f}) shows more disordered staging")
    print(f"\n   Δτ = {abs(tau1 - tau2):.3f}")
    print(f"\n💡 To reproduce: --seed {seed}")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# """
# Trajectory Comparison Inspector.

# Compares two models on the same progression, showing how each orders
# patches along the inferred trajectory. Useful for visualizing the
# difference between high-tau and low-tau models.

# Usage:
#     python compare_trajectories.py
#     python compare_trajectories.py --model1 virchow2 --model2 conch --progression BDC
# """

# import sys
# import argparse
# import urllib.parse
# from pathlib import Path
# import logging
# import numpy as np
# import pandas as pd

# # Plotting & Image Imports
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from PIL import Image
# import fsspec

# # Ensure we can import config from parent directory
# project_root = Path(__file__).resolve().parent.parent
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

# try:
#     import config
# except ImportError:
#     print("❌ Critical Error: Could not import 'config.py'.")
#     sys.exit(1)

# from data.progression_embedding_dataset import (
#     ProgressionEmbeddingDataset,
#     RegistryConfig,
# )
# from analysis.dpt import (
#     DPTConfig,
#     DPTMetric,
#     DPTResult,
#     cohort_to_anndata,
#     compute_dpt,
# )

# logging.basicConfig(level=logging.INFO, format="%(message)s")
# logger = logging.getLogger(__name__)

# # =============================================================================
# # CONFIGURATION - Edit these defaults as needed
# # =============================================================================

# # Default models to compare
# MODEL_HIGH_TAU = "uni2"      # Typically high trajectory fidelity
# MODEL_LOW_TAU = "dinov2"     # Typically lower trajectory fidelity

# # Default progression to analyze
# TARGET_PROGRESSION = "CRC-Serrated"   # Options: "SCC", "CRC-Conventional", "CRC-Serrated", "BDC"

# # Number of patches to sample along trajectory
# NUM_SAMPLES = 10

# # =============================================================================

# EMBEDDING_TYPE = "final_embedding"

# # Stage colors for visual distinction
# STAGE_COLORS = {
#     0: '#2ecc71',  # Green - early/normal
#     1: '#f39c12',  # Orange - intermediate
#     2: '#e74c3c',  # Red - late/invasive
#     3: '#9b59b6',  # Purple - if 4 stages
# }


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Compare DPT trajectories for two models on the same progression."
#     )
#     parser.add_argument(
#         "--model1", 
#         type=str, 
#         default=MODEL_HIGH_TAU,
#         choices=config.EXPECTED_MODELS,
#         help=f"First model, typically high-tau (default: {MODEL_HIGH_TAU})"
#     )
#     parser.add_argument(
#         "--model2", 
#         type=str, 
#         default=MODEL_LOW_TAU,
#         choices=config.EXPECTED_MODELS,
#         help=f"Second model, typically low-tau (default: {MODEL_LOW_TAU})"
#     )
#     parser.add_argument(
#         "--progression", 
#         type=str, 
#         default=TARGET_PROGRESSION,
#         help=f"Name of the progression (default: {TARGET_PROGRESSION})"
#     )
#     parser.add_argument(
#         "--num_samples", 
#         type=int, 
#         default=NUM_SAMPLES,
#         help=f"Number of patches to sample along each trajectory (default: {NUM_SAMPLES})"
#     )
#     return parser.parse_args()


# def compute_trajectory_for_model(
#     prog_config: dict,
#     model_name: str,
#     patch_ids: list,
#     num_samples: int
# ) -> tuple[pd.DataFrame, float]:
#     """
#     Compute DPT trajectory for a single model and return selected patches + tau.
    
#     Returns:
#         Tuple of (selected_patches_df, tau_value)
#     """
#     registry_config = RegistryConfig(
#         bucket=prog_config["bucket"],
#         prefix=prog_config["prefix"],
#         ordered_classes=prog_config["classes"],
#         models=[model_name],
#         progression_name=prog_config["name"],
#         scan_all_models=False
#     )
#     dataset = ProgressionEmbeddingDataset(registry_config)
    
#     tau_value = np.nan
    
#     try:
#         dataset.load_model_into_memory(model_name, patch_ids, EMBEDDING_TYPE)
#         cohort_df = dataset.get_cohort(patch_ids, model_name, EMBEDDING_TYPE)
        
#         if cohort_df.empty:
#             logger.error(f"No data found for {model_name}")
#             return pd.DataFrame(), tau_value
        
#         adata = cohort_to_anndata(cohort_df, prog_config["classes"])
        
#         # Configure DPT with TAU metric enabled
#         dpt_conf = DPTConfig(
#             n_neighbors=config.DPT["n_neighbors"],
#             n_diffusion_components=config.DPT["n_diffusion_components"],
#             metrics={DPTMetric.TAU},  # Request tau calculation
#             subsample_size=2000
#         )
        
#         # Compute DPT and get result with tau
#         dpt_result = compute_dpt(adata, prog_config["root_class"], dpt_conf)
#         tau_value = dpt_result.tau
        
#         logger.info(f"   {model_name}: τ = {tau_value:.3f}")
        
#         if "dpt_pseudotime" not in adata.obs:
#             logger.error(f"DPT failed for {model_name}")
#             return pd.DataFrame(), tau_value
        
#         # Extract and normalize pseudotime to [0, 1]
#         trajectory_df = adata.obs.copy()
#         trajectory_df = trajectory_df[np.isfinite(trajectory_df["dpt_pseudotime"])]
        
#         pt_min = trajectory_df["dpt_pseudotime"].min()
#         pt_max = trajectory_df["dpt_pseudotime"].max()
#         trajectory_df["dpt_pseudotime_norm"] = (
#             (trajectory_df["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)
#         )
        
#         trajectory_df = trajectory_df.sort_values("dpt_pseudotime_norm")
        
#         # Select evenly spaced patches by normalized pseudotime
#         target_values = np.linspace(0, 1, num_samples)
#         pseudotimes = trajectory_df["dpt_pseudotime_norm"].values
#         selected_ilocs = []
        
#         for target in target_values:
#             idx = (np.abs(pseudotimes - target)).argmin()
#             selected_ilocs.append(idx)
        
#         # Keep unique indices in order
#         seen = set()
#         unique_ilocs = []
#         for idx in selected_ilocs:
#             if idx not in seen:
#                 seen.add(idx)
#                 unique_ilocs.append(idx)
        
#         selected_patches = trajectory_df.iloc[unique_ilocs].copy()
#         selected_patches['model'] = model_name
        
#         return selected_patches, tau_value
        
#     finally:
#         dataset.clear_cache()


# def fetch_image(fs, gcs_path: str) -> Image.Image:
#     """Fetch image from GCS, return None if failed."""
#     try:
#         with fs.open(gcs_path, 'rb') as f:
#             return Image.open(f).convert("RGB")
#     except Exception as e:
#         logger.warning(f"Could not load: {gcs_path}: {e}")
#         return None


# def plot_comparison(
#     df1: pd.DataFrame,
#     df2: pd.DataFrame,
#     model1: str,
#     model2: str,
#     tau1: float,
#     tau2: float,
#     prog_config: dict,
#     output_path: Path
# ):
#     """
#     Create side-by-side comparison figure with two rows of patches.
#     """
#     fs = fsspec.filesystem("gcs")
#     bucket = prog_config["bucket"]
#     image_subdir = prog_config.get("image_subdir", "imagenet")
    
#     n1, n2 = len(df1), len(df2)
#     n_cols = max(n1, n2)
    
#     # Figure setup: 2 rows (one per model) + space for labels
#     fig_width = max(12, n_cols * 2.2)
#     fig_height = 6.5
    
#     fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, fig_height))
    
#     # Handle single column edge case
#     if n_cols == 1:
#         axes = axes.reshape(2, 1)
    
#     # Title for the whole figure
#     progression_name = prog_config["name"]
#     fig.suptitle(
#         f"Trajectory Comparison: {progression_name}",
#         fontsize=14,
#         fontweight='bold',
#         y=0.98
#     )
    
#     def plot_row(row_idx, df, model_name, tau_val):
#         """Plot one row of patches."""
#         if np.isfinite(tau_val):
#             tau_str = f"(τ = {tau_val:.2f})"
#         else:
#             tau_str = "(τ = N/A)"
#         row_label = f"{model_name}\n{tau_str}"
        
#         for col_idx in range(n_cols):
#             ax = axes[row_idx, col_idx]
            
#             if col_idx >= len(df):
#                 ax.axis('off')
#                 continue
            
#             row = df.iloc[col_idx]
            
#             # Build GCS path
#             cls = row['class']
#             slide_id = str(row['slide_id'])
#             patch_id = str(row['patch_id'])
            
#             if not slide_id.startswith("slide_"):
#                 slide_id = f"slide_{slide_id}"
#             if not patch_id.startswith("patch_"):
#                 patch_id = f"patch_{patch_id}"
            
#             gcs_path = f"gs://{bucket}/{image_subdir}/train/{cls}/{slide_id}/{patch_id}.png"
            
#             # Fetch and display image
#             img = fetch_image(fs, gcs_path)
#             if img is not None:
#                 ax.imshow(img)
#             else:
#                 ax.text(0.5, 0.5, "Not\nFound", ha='center', va='center', 
#                        fontsize=10, color='red')
#                 ax.set_facecolor('#f0f0f0')
            
#             # Get stage info
#             stage_val = row.get('stage_int', -1)
#             stage = int(stage_val) if pd.notna(stage_val) else -1
#             pt_norm = row.get('dpt_pseudotime_norm', row.get('dpt_pseudotime', 0))
            
#             # Color border by stage
#             border_color = STAGE_COLORS.get(stage, '#95a5a6')
#             for spine in ax.spines.values():
#                 spine.set_visible(True)
#                 spine.set_color(border_color)
#                 spine.set_linewidth(3)
            
#             ax.set_xticks([])
#             ax.set_yticks([])
            
#             # Labels below image
#             if row_idx == 1:  # Bottom row gets pseudotime labels
#                 ax.set_xlabel(f"PT: {pt_norm:.2f}", fontsize=9)
            
#             # Stage label above image
#             stage_label = f"Stage {stage}" if stage >= 0 else "?"
#             ax.set_title(stage_label, fontsize=10, color=border_color, fontweight='bold')
        
#         # Row label on the left
#         axes[row_idx, 0].set_ylabel(
#             row_label, 
#             fontsize=11, 
#             fontweight='bold',
#             rotation=0,
#             ha='right',
#             va='center',
#             labelpad=60
#         )
    
#     # Plot both rows
#     plot_row(0, df1, model1, tau1)
#     plot_row(1, df2, model2, tau2)
    
#     # Add legend for stages
#     stage_names = prog_config.get("classes", [])
#     legend_elements = []
#     for i, name in enumerate(stage_names):
#         color = STAGE_COLORS.get(i, '#95a5a6')
#         # Truncate long class names
#         display_name = (name[:25] + '...') if len(name) > 28 else name
#         legend_elements.append(
#             plt.Line2D([0], [0], marker='s', color='w', 
#                       markerfacecolor=color, markersize=10,
#                       label=f"Stage {i}: {display_name}")
#         )
    
#     fig.legend(
#         handles=legend_elements,
#         loc='lower center',
#         ncol=min(len(stage_names), 4),
#         fontsize=9,
#         frameon=True,
#         bbox_to_anchor=(0.5, 0.02)
#     )
    
#     plt.tight_layout(rect=[0.08, 0.08, 1, 0.95])
#     plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     logger.info(f"✅ Comparison plot saved to {output_path}")


# def main():
#     args = parse_args()
    
#     model1 = args.model1
#     model2 = args.model2
#     target_progression = args.progression
#     num_samples = args.num_samples
    
#     # Find progression config
#     prog_config = next(
#         (p for p in config.PROGRESSIONS if p["name"] == target_progression),
#         None
#     )
    
#     if not prog_config:
#         available = [p["name"] for p in config.PROGRESSIONS]
#         logger.error(f"❌ Progression '{target_progression}' not found.")
#         logger.error(f"   Available: {available}")
#         sys.exit(1)
    
#     logger.info(f"🔍 Comparing Trajectories on {target_progression}")
#     logger.info(f"   Model 1 (high-tau expected): {model1}")
#     logger.info(f"   Model 2 (low-tau expected):  {model2}")
    
#     # Get shared patch IDs (same patches for both models)
#     registry_config = RegistryConfig(
#         bucket=prog_config["bucket"],
#         prefix=prog_config["prefix"],
#         ordered_classes=prog_config["classes"],
#         models=[model1],  # Just need one to get patch IDs
#         progression_name=target_progression,
#         scan_all_models=False
#     )
#     dataset = ProgressionEmbeddingDataset(registry_config)
    
#     patch_ids = dataset.sample_patch_ids(
#         n_per_class=config.EVALUATION["n_per_class"],
#         max_per_slide=config.EVALUATION["max_per_slide"],
#         seed=config.EVALUATION["seed"]
#     )
#     dataset.clear_cache()
    
#     logger.info(f"   Using {len(patch_ids)} shared patches\n")
    
#     # Compute trajectories for both models (now returns tau)
#     logger.info(f"⚙️  Computing trajectory for {model1}...")
#     df1, tau1 = compute_trajectory_for_model(prog_config, model1, patch_ids, num_samples)
    
#     logger.info(f"⚙️  Computing trajectory for {model2}...")
#     df2, tau2 = compute_trajectory_for_model(prog_config, model2, patch_ids, num_samples)
    
#     if df1.empty or df2.empty:
#         logger.error("❌ Failed to compute one or both trajectories.")
#         sys.exit(1)
    
#     # Print summary
#     print("\n" + "=" * 80)
#     print(f"TRAJECTORY COMPARISON: {target_progression}")
#     print("=" * 80)
#     print(f"\n{'Model':<15} | {'Kendall τ':<12} | {'Interpretation'}")
#     print("-" * 50)
#     print(f"{model1:<15} | {tau1:<12.3f} | {'Higher fidelity' if tau1 > tau2 else 'Lower fidelity'}")
#     print(f"{model2:<15} | {tau2:<12.3f} | {'Higher fidelity' if tau2 > tau1 else 'Lower fidelity'}")
    
#     for model_name, df, tau in [(model1, df1, tau1), (model2, df2, tau2)]:
#         print(f"\n{model_name} (τ = {tau:.3f}):")
#         print(f"  {'PT (norm)':<10} | {'Stage':<6} | {'Class'}")
#         print(f"  {'-'*50}")
#         for _, row in df.iterrows():
#             pt = row.get('dpt_pseudotime_norm', row.get('dpt_pseudotime', 0))
#             stage = int(row['stage_int']) if pd.notna(row.get('stage_int')) else '?'
#             cls = row['class'][:40]
#             print(f"  {pt:<10.3f} | {stage:<6} | {cls}")
    
#     print("\n" + "=" * 80 + "\n")
    
#     # Generate comparison plot
#     plot_filename = f"comparison_{model1}_vs_{model2}_{target_progression.replace(' ', '_')}.png"
#     plot_path = config.RESULTS_DIR / "plots" / plot_filename
#     plot_path.parent.mkdir(parents=True, exist_ok=True)
    
#     plot_comparison(
#         df1, df2,
#         model1, model2,
#         tau1, tau2,
#         prog_config,
#         plot_path
#     )
    
#     # Print conclusion
#     if tau1 > tau2:
#         better, worse = model1, model2
#         better_tau, worse_tau = tau1, tau2
#     else:
#         better, worse = model2, model1
#         better_tau, worse_tau = tau2, tau1
    
#     print(f"📊 Summary:")
#     print(f"   {better} (τ={better_tau:.2f}) better preserves disease progression order")
#     print(f"   {worse} (τ={worse_tau:.2f}) shows more disordered staging")
#     print(f"\n   Δτ = {abs(tau1 - tau2):.3f}")


# if __name__ == "__main__":
#     main()