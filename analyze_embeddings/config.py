import os
from datetime import datetime
from typing import List, Dict, TypedDict
from pathlib import Path

RESULTS_DIR = Path("results")

# ── Active run ────────────────────────────────────────────────────────────────
# Update this after a successful data generation run.
# Both results and plots directories are derived from this single value.
RESULTS_RUN = "2025-01-01_000000"

RESULTS_RUN_DIR = RESULTS_DIR / RESULTS_RUN
PLOTS_OUTPUT_DIR = Path("plots") / RESULTS_RUN

FULL_RESULTS_OUTPUT_PATH = RESULTS_RUN_DIR / "full_manifold_evaluation.csv"
NULL_RESULTS_OUTPUT_PATH = RESULTS_RUN_DIR / "null_manifold_evaluation.csv"
PERMUTATION_RESULTS_OUTPUT_PATH = RESULTS_RUN_DIR / "stage_permutation_specificity.csv"
GENERALIZABILITY_RESULTS_OUTPUT_PATH = RESULTS_RUN_DIR / "generalizability_results.csv"


def get_output_dir() -> Path:
    """Get today's output directory inside RESULTS_DIR.

    Pattern: results/YYYY-MM-DD_HHMMSS/

    Data generation scripts call this to create a timestamped output directory.
    After a successful run, update RESULTS_RUN above to point plotting scripts
    at the new data.

    Returns:
        Path to today's output directory.
    """
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = RESULTS_DIR / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
class ProgressionConfig(TypedDict):
    name: str
    bucket: str
    prefix: str
    classes: List[str]
    root_class: str 
    image_subdir: str

# Define the models you expect to see. 
EXPECTED_MODELS = [
    "virchow2",
    "uni2",
    "conch",
    "gigapath",
    "musk",
    "dinov2"
]

# Define the progressions
PROGRESSIONS: List[ProgressionConfig] = [
    {
        "name": "SCC",
        "bucket": "spider-skin",
        "prefix": "embeddings/computed",
        "classes": [
            "Epidermis",
            "Actinic keratosis",
            "Carcinoma in situ",
            "Squamous cell carcinoma"
        ],
        "root_class": "Epidermis",
        "image_subdir": "imagenet_context1"
    },
    {
        "name": "CRC-Conventional",
        "bucket": "spider-colorectal",
        "prefix": "embeddings/computed",
        "classes": [
            "Adenoma low grade",
            "Adenoma high grade",
            "Adenocarcinoma low grade",
            "Adenocarcinoma high grade"
        ],
        "root_class": "Adenoma low grade",
        "image_subdir": "imagenet"
    },
    {
        "name": "CRC-Serrated",
        "bucket": "spider-colorectal",
        "prefix": "embeddings/computed",
        "classes": [
            "Hyperplastic polyp",
            "Sessile serrated lesion",
            "Adenocarcinoma high grade"
        ],
        "root_class": "Hyperplastic polyp",
        "image_subdir": "imagenet"
    },
    {
        "name": "BDC",
        "bucket": "spider-breast",
        "prefix": "embeddings/computed",
        "classes": [
            "Ductal carcinoma in situ (low-grade)",
            "Ductal carcinoma in situ (high-grade)",
            "Invasive non-special type carcinoma"
        ],
        "root_class": "Ductal carcinoma in situ (low-grade)",
        "image_subdir": "imagenet-central"
    }
]


def get_progressions(names: List[str] = None) -> List[ProgressionConfig]:
    """Return progression configs filtered by name. If names is None, return all."""
    if names is None:
        return PROGRESSIONS
    available = {p["name"]: p for p in PROGRESSIONS}
    result = []
    for name in names:
        if name not in available:
            raise ValueError(
                f"Unknown progression '{name}'. Available: {list(available.keys())}"
            )
        result.append(available[name])
    return result


EVALUATION = {
    "n_per_class": 1000,
    "max_per_slide": 50,
    "seed": 42,
    "embedding_type": "final_embedding",
    # evaluate_dpt: bootstrap resampling iterations for metric confidence intervals
    "n_bootstrap_ci": 100,
    # generate_null_distribution: label-shuffle permutations for null tau distribution
    "n_null_permutations": 10,
    # evaluate_generalizability: repeated few-shot sampling trials per shot count
    "n_fewshot_trials": 10,
    # evaluate_generalizability: shots per class for few-shot probing
    "fewshot_n_shots": [5, 10, 20],
}

TEST_EVALUATION = {
    "n_per_class": 50,
    "max_per_slide": 10,
    "seed": 42,
    "embedding_type": "final_embedding",
    "n_bootstrap_ci": 2,
    "n_null_permutations": 2,
    "n_fewshot_trials": 2,
    "fewshot_n_shots": [5],
}

DPT = {
    "n_neighbors": 100,
    "n_diffusion_components": 10,
}

PATCH_EMBEDDINGS = [
    "patch_mean_0.125",
    "patch_mean_0.375",
    "patch_mean_0.625",
    "patch_mean_0.875",
    "patch_mean_1.0",
]

CLS_EMBEDDINGS = [
    "cls_0.125",
    "cls_0.375",
    "cls_0.625",
    "cls_0.875",
    "cls_1.0",
]

REGISTER_EMBEDDINGS = [
    "register_mean_0.125",
    "register_mean_0.375",
    "register_mean_0.625",
    "register_mean_0.875",
    "register_mean_1.0",
]

FINAL_EMBEDDING = "final_embedding"