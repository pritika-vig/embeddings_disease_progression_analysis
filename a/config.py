"""
Configuration for Pathology Progression Analysis.
Centralizes all experimental parameters, paths, and disease definitions.
"""

# ==============================================================================
# 1. EXPERIMENT CONTROL
# ==============================================================================
# The specific progression to run in this execution
CURRENT_PROGRESSION = "breast_ductal"

# ==============================================================================
# 2. GCS & DATA PATHS
# ==============================================================================
GCS_SETTINGS = {
    "bucket_name": "spider-breast",
    "prefix": "embeddings/computed",
    "reference_model": "virchow2",  # The model used to define the frozen cohort
    "project": None                 # Optional GCS project ID
}

# ==============================================================================
# 3. SAMPLING STRATEGY
# ==============================================================================
SAMPLING_SETTINGS = {
    "patches_per_class": 1000,  # Target N patches per stage
    "per_slide_cap": 50,        # Max patches per slide (prevents slide dominance)
    "random_state": 35          # Seed for reproducibility
}

# ==============================================================================
# 4. ANALYSIS PARAMETERS
# ==============================================================================
ANALYSIS_SETTINGS = {
    "layer_key": "final_embedding", # The embedding layer to evaluate
    "n_neighbors": 100,              # k-NN graph size for Diffusion Map
    "n_bootstraps": 50              # Number of bootstrap iterations for CI
}

# ==============================================================================
# 5. MODEL REGISTRY
# ==============================================================================
MODELS_TO_EVALUATE = [
    "virchow2",
    "uni2",
    "gigapath",
    "conch",
    "musk",
    "dinov2"
]

# ==============================================================================
# 6. PROGRESSION DEFINITIONS
# ==============================================================================
# Maps internal keys to ordered lists of class labels found in the parquet files.
PROGRESSIONS = {
    "skin_scc": [
        "Epidermis",
        "Actinic keratosis",
        "Squamous cell carcinoma in situ",
        "Invasive squamous cell carcinoma"
    ],
    "colorectal_conventional": [
        "Adenoma (low grade)",
        "Adenoma (high grade)",
        "Adenocarcinoma (low grade)",
        "Adenocarcinoma (high grade)"
    ],
    "colorectal_serrated": [
        "Hyperplastic polyp",
        "Sessile serrated lesion",
        "Adenocarcinoma (high grade)"
    ],
    "breast_ductal": [
        "Ductal carcinoma in situ (low-grade)",
        "Ductal carcinoma in situ (high-grade)",
        "Invasive non-special type carcinoma"
    ]
}