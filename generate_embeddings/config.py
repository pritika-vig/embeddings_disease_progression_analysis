from dataclasses import dataclass
from typing import List
from pathlib import Path

# ===============================
# Output Configuration
# ===============================
class OutputConfig:
    # Production path
    PROD_PATH = "embeddings/computed"
    
    # Test path
    TEST_PATH = "embeddings/computed/test_run"

    @staticmethod
    def get_path(test_run: bool) -> str:
        return OutputConfig.TEST_PATH if test_run else OutputConfig.PROD_PATH

# ===============================
# Constants
# ===============================
DATA_ROOT = Path("/home/pritika_l_vig_gmail_com/data")
SHARD_SIZE = 10000
LAYER_FRACTIONS = [0.125, 0.375, 0.625, 0.875, 1.0]

@dataclass
class DatasetConfig:
    dataset_name: str
    classes: List[str]
    
    @property
    def bucket(self) -> str:
        bucket_map = {
            "spider-breast": "spider-breast",
            "spider-col": "spider-colorectal",
            "spider-skin": "spider-skin"
        }
        return bucket_map.get(self.dataset_name)

# ===============================
# Datasets
# ===============================
DATASETS = [
    DatasetConfig(
        dataset_name="spider-breast",
        classes=[
            "Ductal carcinoma in situ (low-grade)",
            "Ductal carcinoma in situ (high-grade)", 
            "Invasive non-special type carcinoma"
        ]
    ),
    DatasetConfig(
        dataset_name="spider-col",
        classes=[
            "Adenoma low grade", "Adenoma high grade", 
            "Adenocarcinoma low grade", "Adenocarcinoma high grade", 
            "Hyperplastic polyp", "Sessile serrated lesion"
        ]
    ),
    DatasetConfig(
        dataset_name="spider-skin",
        classes=[
            "Actinic keratosis", "Epidermis", 
            "Carcinoma in situ", "Squamous cell carcinoma"
        ]
    )
]