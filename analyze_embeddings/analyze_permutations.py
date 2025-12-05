import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau

import config
from analysis.dpt import DPTConfig, DPTMetric, cohort_to_anndata, compute_dpt
from data.progression_embedding_dataset import ProgressionEmbeddingDataset, RegistryConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_permutation_test(progression_name, model_name):
    """
    Tests all permutations of stage ordering for a single best model.
    """
    prog = next(p for p in config.PROGRESSIONS if p["name"] == progression_name)
    true_order = prog["classes"]
    
    # Setup Dataset
    registry_config = RegistryConfig(
        bucket=prog["bucket"], prefix=prog["prefix"], ordered_classes=true_order,
        models=config.EXPECTED_MODELS, progression_name=progression_name, scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    dpt_config = DPTConfig(
        n_neighbors=50, n_diffusion_components=15, metrics={DPTMetric.TAU}, subsample_size=2000
    )

    # Load Data
    patch_ids = dataset.sample_patch_ids(
        n_per_class=config.EVALUATION["n_per_class"], 
        max_per_slide=config.EVALUATION["max_per_slide"], seed=42
    )
    dataset.load_model_into_memory(model_name, patch_ids)
    
    # We use CLS token usually for best ordering, or final_embedding if you prefer
    cohort_df = dataset.get_cohort(patch_ids, model_name, embedding_type="final_embedding")
    
    # 1. Compute DPT Geometry ONCE using the True Root
    # The manifold shape is fixed by the data. 
    adata = cohort_to_anndata(cohort_df, true_order)
    res = compute_dpt(adata, prog["root_class"], dpt_config)
    
    if not res.is_valid:
        raise ValueError("DPT failed on true ordering")
        
    # Extract Pseudotime (X) and True Integer Stages (Y)
    valid_mask = np.isfinite(adata.obs["dpt_pseudotime"])
    pt_values = adata.obs["dpt_pseudotime"][valid_mask].values
    
    # Get the raw string labels for the patches
    patch_labels = adata.obs["class"][valid_mask].values
    
    # 2. Iterate ALL Permutations of the classes
    results = []
    perms = list(itertools.permutations(true_order))
    
    logger.info(f"Testing {len(perms)} permutations...")
    
    for perm in perms:
        # Create a new mapping for this specific permutation
        # e.g. if perm is ('Invasive', 'Epidermis', ...), Invasive becomes 0
        perm_map = {label: i for i, label in enumerate(perm)}
        
        # Map the patch string labels to these new integers
        permuted_integers = np.array([perm_map[l] for l in patch_labels])
        
        # Calculate Correlation against the FIXED manifold time
        t, _ = kendalltau(pt_values, permuted_integers)
        
        is_true_order = (perm == tuple(true_order))
        is_reverse_order = (perm == tuple(true_order[::-1]))
        
        results.append({
            "Permutation": " -> ".join(perm),
            "Tau": t,
            "Type": "True Order" if is_true_order else ("Reverse" if is_reverse_order else "Wrong Order")
        })
        
    return pd.DataFrame(results)

def plot_permutation_histogram(df, prog_name):
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    plt.figure(figsize=(10, 6))
    
    # Split data
    true_val = df[df["Type"] == "True Order"]["Tau"].values[0]
    wrong_vals = df[df["Type"] == "Wrong Order"]["Tau"].values
    
    # Plot Distribution of Wrong Orders
    sns.histplot(wrong_vals, kde=True, color="gray", stat="density", alpha=0.4, label="Incorrect Permutations")
    
    # Add True Order Line
    plt.axvline(true_val, color="#d62728", linestyle='-', linewidth=3, label=f"True Biological Order (τ={true_val:.2f})")
    
    # Formatting
    plt.title(f"Validation of Biological Progression: {prog_name}")
    plt.xlabel("Kendall's Tau with Manifold Pseudotime")
    plt.ylabel("Density")
    plt.legend(loc="upper left")
    
    # Add simple text annotation for p-value proxy
    rank = sum(df["Tau"] >= true_val)
    total = len(df)
    plt.text(0.05, 0.95, f"Rank: {rank}/{total}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"permutation_test_{prog_name}.png", dpi=300)
    print(f"Saved permutation_test_{prog_name}.png")

if __name__ == "__main__":
    # You can pick your best model here, e.g., 'uni2' or 'virchow2'
    df = run_permutation_test("SCC", model_name="virchow2")
    plot_permutation_histogram(df, "SCC")
    
    # Print the top 5 orderings to see what the model thinks is 'close'
    print("\nTop 5 Orderings found by Manifold:")
    print(df.sort_values("Tau", ascending=False).head(5)[["Permutation", "Tau"]])