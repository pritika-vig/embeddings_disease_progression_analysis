from typing import List, Dict, TypedDict
from pathlib import Path

RESULTS_DIR = Path("results")
PLOTS_OUTPUT_DIR = Path("plots")
FULL_RESULTS_OUTPUT_PATH = RESULTS_DIR / "full_manifold_evaluation_100.csv"
NULL_RESULTS_OUTPUT_PATH = RESULTS_DIR / "null_manifold_evaluation.csv"
PERMUTATION_RESULTS_OUTPUT_PATH = RESULTS_DIR / "stage_permutation_specificity.csv"
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

EVALUATION = {
    "n_per_class": 1000,
    "max_per_slide": 50,
    "n_bootstrap": 10,
    "seed": 42,
    "embedding_type": "final_embedding",
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