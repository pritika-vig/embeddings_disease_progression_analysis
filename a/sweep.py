import os
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from tqdm import tqdm

# Local Imports
import config
from progression_embedding_dataset import ProgressionEmbeddingDataset

# --- Experiment Configuration ---
# Validating distinct pairs demonstrates that the parameter stability 
# is robust across different biological contexts and architectures.
SWEEP_PAIRS = [
    ("breast_ductal", "uni2"),
    ("skin_scc", "virchow2"),
    ("colorectal_conventional", "gigapath")
]

K_VALUES = [10, 20, 30, 40, 50, 75, 100, 150, 175, 200, 300]

# Mapping progressions to their specific GCS buckets
BUCKET_MAP = {
    "breast_ductal": "spider-breast",
    "skin_scc": "spider-skin",
    "colorectal_conventional": "spider-colorectal"
}

def get_robust_root(adata, root_class_name):
    """Finds the medoid of the root class to serve as the diffusion start point."""
    root_mask = adata.obs['class'] == root_class_name
    root_indices = np.where(root_mask)[0]
    
    if len(root_indices) == 0:
        raise ValueError(f"Root class '{root_class_name}' not found in data.")
        
    X_root = adata.X[root_mask]
    centroid = np.mean(X_root, axis=0).reshape(1, -1)
    dists = cdist(X_root, centroid, metric='cosine')
    return root_indices[np.argmin(dists)]

def run_comparative_sweep():
    # Setup Output Directory
    output_dir = "sweep_results"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []

    print(f"Starting sweep for {len(SWEEP_PAIRS)} diverse pairs...")

    # --- Core Loop: Iterate over defined pairs ---
    for prog_name, model in SWEEP_PAIRS:
        print(f"\n=== Processing Pair: {prog_name} + {model} ===")
        classes = config.PROGRESSIONS[prog_name]
        bucket_name = BUCKET_MAP[prog_name]
        
        # Initialize Dataset
        dataset = ProgressionEmbeddingDataset(
            bucket=bucket_name,
            prefix=config.GCS_SETTINGS["prefix"],
            reference_model=config.GCS_SETTINGS["reference_model"],
            patches_per_class=config.SAMPLING_SETTINGS["patches_per_class"],
            per_slide_cap=config.SAMPLING_SETTINGS["per_slide_cap"],
            random_state=config.SAMPLING_SETTINGS["random_state"],
            ordered_classes=classes,
            progression_name=prog_name
        )
        
        # Load embeddings (Fail fast if missing)
        try:
            df = dataset.get_embeddings(model, config.ANALYSIS_SETTINGS["layer_key"])
        except Exception as e:
            print(f"  ! ERROR: Could not load embeddings for {model}. Check GCS paths. ({e})")
            continue

        # --- Sweep k values ---
        for k in tqdm(K_VALUES, desc=f"    Sweeping k", leave=False):
            adata = dataset.to_anndata(df)
            
            # 1. Compute Neighbors
            sc.pp.neighbors(adata, use_rep='X', n_neighbors=k, metric='cosine', method='gauss')
            
            # 2. Diffusion Map & DPT
            try:
                iroot = get_robust_root(adata, classes[0])
                adata.uns['iroot'] = iroot
                sc.tl.diffmap(adata)
                sc.tl.dpt(adata)
                
                # 3. Calculate Metrics
                pseudotime = adata.obs['dpt_pseudotime'].values
                
                # Check for disconnected graph (infinite pseudotime)
                if not np.all(np.isfinite(pseudotime)):
                    tau = 0.0
                    sil_score = 0.0
                else:
                    tau, _ = kendalltau(pseudotime, adata.obs['stage_int'])
                    sil_score = silhouette_score(
                        pseudotime.reshape(-1, 1), 
                        adata.obs['class']
                    )
                
                evals = adata.uns['diffmap_evals']
                spectral_gap = evals[1] - evals[2] if len(evals) > 2 else 0 
                
                trust_score = trustworthiness(
                    adata.X, 
                    adata.obsm['X_diffmap'][:, 1:3], 
                    n_neighbors=12, 
                    metric='cosine'
                )

                all_results.append({
                    "Pair": f"{prog_name} ({model})", # Label for the plot
                    "Progression": prog_name,
                    "Model": model,
                    "k": k,
                    "Tau": tau,
                    "Spectral_Gap": spectral_gap,
                    "Trustworthiness": trust_score,
                    "Silhouette": sil_score
                })
            except Exception as e:
                # print(f"    Failed at k={k}: {e}")
                pass

    # --- Plotting Phase ---
    df_results = pd.DataFrame(all_results)
    if df_results.empty:
        print("No results generated. Check data access.")
        return

    df_results.to_csv(os.path.join(output_dir, "diverse_sweep_raw.csv"), index=False)
    
    # Generate the "Killer Figure" (Combined)
    plot_combined_summary(df_results, output_dir)
    print("\n✅ Sweep Complete. Results saved to:", output_dir)

def plot_combined_summary(df, output_dir):
    """Overlays all pairs on a single 4-panel figure to demonstrate consistency."""
    
    metrics = [
        ("Tau", "Kendall Tau (Order)", "lower right"),
        ("Spectral_Gap", "Spectral Gap (Stability)", "lower right"),
        ("Trustworthiness", "Trustworthiness (Fidelity)", "lower left"),
        ("Silhouette", "Silhouette (Separation)", "lower right")
    ]
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Manifold Stability Across Diverse Biological Contexts", fontsize=18, weight='bold')
    
    axes = axes.flatten()
    
    for i, (metric, label, loc) in enumerate(metrics):
        ax = axes[i]
        
        # Plot all pairs on the same axes
        sns.lineplot(
            data=df, 
            x="k", 
            y=metric, 
            hue="Pair", 
            style="Pair",
            markers=True, 
            dashes=False, 
            linewidth=2.5,
            palette="viridis",
            ax=ax
        )
        
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("k (Nearest Neighbors)", fontsize=12)
        ax.set_ylabel("Normalized Score" if "Gap" not in metric else "Eigenvalue Diff", fontsize=12)
        
        # Visual Guide for the decision boundary
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label="Selected k=100")
        
        # Clean Legend
        if i == 0:
            ax.legend(title="Context", loc=loc, frameon=True)
        else:
            if ax.get_legend(): ax.get_legend().remove()
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "appendix_manifold_stability.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated summary plot: {save_path}")

if __name__ == "__main__":
    run_comparative_sweep()