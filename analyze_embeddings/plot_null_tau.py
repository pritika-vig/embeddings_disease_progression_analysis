import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import kendalltau

import config
from data.progression_embedding_dataset import ProgressionEmbeddingDataset, RegistryConfig
from analysis.dpt import DPTConfig, DPTMetric, cohort_to_anndata, compute_dpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_null_tau(adata, n_shuffles=50):
    """
    Compute Tau distribution when labels are randomly shuffled across patches.
    This breaks the relationship between manifold structure and biology.
    """
    # Keep valid DPT values
    valid_mask = np.isfinite(adata.obs["dpt_pseudotime"])
    if valid_mask.sum() < 10: return np.nan
    
    pt = adata.obs["dpt_pseudotime"][valid_mask].values
    true_stages = adata.obs["stage_int"][valid_mask].values
    
    null_taus = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_shuffles):
        # Shuffle the stage labels
        shuffled_stages = rng.permutation(true_stages)
        t, _ = kendalltau(pt, shuffled_stages)
        null_taus.append(t)
        
    return np.mean(null_taus), np.std(null_taus)

def run_null_experiment(progression_name):
    # Setup
    prog = next(p for p in config.PROGRESSIONS if p["name"] == progression_name)
    registry_config = RegistryConfig(
        bucket=prog["bucket"],
        prefix=prog["prefix"],
        ordered_classes=prog["classes"],
        models=config.EXPECTED_MODELS,
        progression_name=progression_name,
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    
    # Config: Only compute Tau
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

    rows = []
    
    for model in config.EXPECTED_MODELS:
        logger.info(f"Processing {model}...")
        try:
            dataset.load_model_into_memory(model, patch_ids)
            
            # Fetch 'final_embedding' (Layer 1.0)
            # You can change this to 'cls_1.0' if that performed better in your layer scan
            cohort_df = dataset.get_cohort(patch_ids, model, embedding_type="final_embedding")
            adata = cohort_to_anndata(cohort_df, prog["classes"])
            
            # 1. Compute True Model Tau (with Bootstrap)
            # We do a mini-bootstrap here just for the error bars
            boot_taus = []
            for i in range(50): # 50 bootstraps for CI
                b_ids = dataset.bootstrap_patch_ids(patch_ids, seed=i)
                # Reconstruct AnnData from cached dataframe (fast)
                b_df = cohort_df[cohort_df['patch_id'].isin(b_ids)] # simplistic bootstrap filtering
                b_adata = cohort_to_anndata(b_df, prog["classes"])
                res = compute_dpt(b_adata, prog["root_class"], dpt_config)
                if res.is_valid: boot_taus.append(res.tau)
            
            model_mean = np.mean(boot_taus)
            model_ci = np.percentile(boot_taus, [2.5, 97.5])
            
            # 2. Compute Null Tau (Random Labels)
            # We compute DPT once on the original data, then shuffle labels
            real_res = compute_dpt(adata, prog["root_class"], dpt_config)
            null_mean, null_std = get_null_tau(adata)
            
            rows.append({
                "Model": model,
                "Condition": "Learned Manifold",
                "Tau": model_mean,
                "Error_Low": model_mean - model_ci[0],
                "Error_High": model_ci[1] - model_mean
            })
            
            rows.append({
                "Model": model,
                "Condition": "Shuffled Labels (Null)",
                "Tau": null_mean,
                "Error_Low": 1.96 * null_std, # 95% CI for Null
                "Error_High": 1.96 * null_std
            })
            
        except Exception as e:
            logger.error(f"Failed {model}: {e}")
        finally:
            dataset.clear_cache()
            
    return pd.DataFrame(rows)

def plot_comparison(df, prog_name):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.figure(figsize=(10, 6))
    
    # Create Bar Plot
    # We define error bars manually using yerr
    
    # Pivot for easier plotting if needed, or just iterate
    models = df["Model"].unique()
    x = np.arange(len(models))
    width = 0.35
    
    learned = df[df["Condition"] == "Learned Manifold"].set_index("Model").reindex(models)
    null = df[df["Condition"] == "Shuffled Labels (Null)"].set_index("Model").reindex(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, learned["Tau"], width, label='Learned Manifold', 
                    yerr=[learned["Error_Low"], learned["Error_High"]], capsize=5, color="#1f77b4", alpha=0.9)
    rects2 = ax.bar(x + width/2, null["Tau"], width, label='Null (Shuffled)', 
                    yerr=[null["Error_Low"], null["Error_High"]], capsize=5, color="#bababa", alpha=0.9)

    ax.set_ylabel("Kendall's Tau")
    ax.set_title(f"Manifold Validity vs Chance: {prog_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.8)
    
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(f"null_comparison_{prog_name}.png", dpi=300)
    print(f"Saved null_comparison_{prog_name}.png")

if __name__ == "__main__":
    df = run_null_experiment("SCC")
    plot_comparison(df, "SCC")