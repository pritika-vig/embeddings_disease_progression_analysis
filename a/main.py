import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Local Imports
import config
from progression_embedding_dataset import ProgressionEmbeddingDataset

def get_robust_root(adata, root_class_name):
    """
    Finds the 'Medoid' patch of the root class.
    Uses the raw X (embeddings) to calculate the centroid.
    """
    # 1. Get all patches in the root class
    root_mask = adata.obs['class'] == root_class_name
    root_indices = np.where(root_mask)[0]
    
    if len(root_indices) == 0:
        return None
    
    # 2. Extract embeddings
    # Since we are not doing PCA, we use .X directly
    X_root = adata.X[root_mask]
        
    # 3. Calculate Centroid
    centroid = np.mean(X_root, axis=0).reshape(1, -1)
    
    # 4. Find the patch closest to the centroid (Medoid)
    # We use cosine distance here to match our graph metric
    dists = cdist(X_root, centroid, metric='cosine')
    medoid_rel_idx = np.argmin(dists)
    
    return root_indices[medoid_rel_idx]

def calculate_dpt_correlation(adata, root_class_name, n_neighbors):
    """
    Runs DPT using Cosine metric on raw embeddings (No PCA).
    """
    try:
        # 1. Compute Manifold (Neighbors)
        sc.pp.neighbors(
            adata, 
            use_rep='X', 
            n_neighbors=n_neighbors, 
            metric='cosine',
            method='gauss'
        )
        
        # 2. Set Robust Root
        iroot = get_robust_root(adata, root_class_name)
        if iroot is None:
            return np.nan
        adata.uns['iroot'] = iroot

        # 3. Run Diffusion Map
        # Note: Scanpy implements Haghverdi 2016 (alpha=1.0) by default logic
        sc.tl.diffmap(adata)

        # 4. Run Diffusion Pseudotime
        sc.tl.dpt(adata)

        # 5. Correlation
        tau, p_val = kendalltau(adata.obs['dpt_pseudotime'], adata.obs['stage_int'])
        return tau

    except Exception as e:
        # Catch "disconnected component" errors gracefully
        # print(f"    [DPT Error]: {e}")
        return np.nan

def run_evaluation():
    # 1. Load Experiment Configuration
    print("========================================")
    print("  PATHOLOGY FOUNDATION MODEL BENCHMARK  ")
    print("  (Scanpy 'Last Stand' Configuration)   ")
    print("========================================")
    
    prog_name = config.CURRENT_PROGRESSION
    ordered_classes = config.PROGRESSIONS[prog_name]
    
    print(f"Progression: {prog_name}")
    print(f"Stages: {ordered_classes}")

    # 2. Initialize Dataset
    dataset = ProgressionEmbeddingDataset(
        bucket=config.GCS_SETTINGS["bucket_name"],
        prefix=config.GCS_SETTINGS["prefix"],
        reference_model=config.GCS_SETTINGS["reference_model"],
        gcs_project=config.GCS_SETTINGS["project"],
        patches_per_class=config.SAMPLING_SETTINGS["patches_per_class"],
        per_slide_cap=config.SAMPLING_SETTINGS["per_slide_cap"],
        random_state=config.SAMPLING_SETTINGS["random_state"],
        ordered_classes=ordered_classes,
        progression_name=prog_name
    )
    
    results = []
    
    # Hyperparam override for this run
    # 40-50 neighbors helps fix disconnected components in high-dim space
    ROBUST_K = 50 
    
    # 3. Iterate Models
    for model in config.MODELS_TO_EVALUATE:
        print(f"\n--- Evaluating Model: {model.upper()} ---")
        
        try:
            # A. Load Data
            df_main = dataset.get_embeddings(
                model=model, 
                layer_key=config.ANALYSIS_SETTINGS["layer_key"]
            )
            
            # B. Main DPT Calculation
            adata_main = dataset.to_anndata(df_main)
            
            main_tau = calculate_dpt_correlation(
                adata_main, 
                root_class_name=ordered_classes[0],
                n_neighbors=ROBUST_K
            )
            print(f"   Kendall's Tau (Main): {main_tau:.4f}")

            # C. Bootstrapping
            n_boot = config.ANALYSIS_SETTINGS["n_bootstraps"]
            print(f"   Running {n_boot} bootstraps...")
            
            boot_samples = dataset.get_bootstrapped_samples(df_main, n_bootstraps=n_boot)
            boot_taus = []
            
            for boot_df in tqdm(boot_samples, desc="   Bootstrapping", leave=False):
                boot_adata = dataset.to_anndata(boot_df)
                t = calculate_dpt_correlation(
                    boot_adata, 
                    root_class_name=ordered_classes[0],
                    n_neighbors=ROBUST_K
                )
                if not np.isnan(t):
                    boot_taus.append(t)
            
            # D. Confidence Intervals
            ci_lower = np.percentile(boot_taus, 2.5) if boot_taus else np.nan
            ci_upper = np.percentile(boot_taus, 97.5) if boot_taus else np.nan
            
            results.append({
                "Model": model,
                "Tau": main_tau,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
                "Progression": prog_name
            })
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model}: {e}")
            results.append({
                "Model": model, "Tau": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan, "Progression": prog_name
            })

    # 4. Final Reporting
    print("\n\n================ RESULTS ================")
    results_df = pd.DataFrame(results)
    print(results_df.sort_values("Tau", ascending=False).to_markdown(index=False, floatfmt=".4f"))
    
    filename = f"results_{prog_name}_scanpy_robust.csv"
    results_df.to_csv(filename, index=False)
    print(f"\nSaved results to {filename}")

if __name__ == "__main__":
    run_evaluation()