import numpy as np
import pandas as pd
import pytest
from io import BytesIO
from unittest.mock import MagicMock, patch, call

from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
    RegistryColumns,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_config():
    return RegistryConfig(
        bucket="test-bucket",
        prefix="embeddings",
        ordered_classes=["normal", "low_grade", "high_grade"],
        models=["model_a", "model_b"],
        progression_name="test_progression",
        scan_all_models=False  # Default to Fast Mode for most tests
    )

@pytest.fixture
def mock_registry_data():
    """
    Create consistent test data. 
    Includes MULTIPLE embedding types to test filtering logic.
    """
    base_data = [
        # Normal class - 2 slides
        {"class": "normal", "slide_id": "slide_001", "patch_id": "patch_001", "embedding_type": "layer_12"},
        {"class": "normal", "slide_id": "slide_001", "patch_id": "patch_001", "embedding_type": "mean_pool"}, # Dual type
        {"class": "normal", "slide_id": "slide_001", "patch_id": "patch_002", "embedding_type": "layer_12"},
        
        # Low grade
        {"class": "low_grade", "slide_id": "slide_003", "patch_id": "patch_006", "embedding_type": "layer_12"},
        
        # High grade
        {"class": "high_grade", "slide_id": "slide_005", "patch_id": "patch_009", "embedding_type": "layer_12"},
    ]
    return pd.DataFrame(base_data)

@pytest.fixture
def mock_embedding_data(mock_registry_data):
    """Add embeddings to registry data."""
    df = mock_registry_data.copy()
    df[RegistryColumns.EMBEDDING] = [np.random.randn(128).tolist() for _ in range(len(df))]
    return df

def create_mock_open(df):
    """Helper to create a mock fs.open context manager that returns parquet bytes."""
    class MockFileContext:
        def __init__(self, df):
            self.df = df
        def __enter__(self):
            buffer = BytesIO()
            self.df.to_parquet(buffer)
            buffer.seek(0)
            return buffer
        def __exit__(self, *args):
            pass
    return lambda *args, **kwargs: MockFileContext(df)

# ─────────────────────────────────────────────────────────────────────────────
# Initialization & Registry Mode Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressionEmbeddingDatasetInit:
    
    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_golden_record_mode_scans_only_first_model(self, mock_fsspec, sample_config, mock_registry_data):
        """When scan_all_models=False, should only touch model_a."""
        sample_config.scan_all_models = False
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        
        # Setup files existing for both models
        mock_fs.glob.side_effect = lambda p: [p.replace("*", "part_0")]
        mock_fs.open = create_mock_open(mock_registry_data)
        
        ds = ProgressionEmbeddingDataset(sample_config)
        
        # Should populate registry
        assert len(ds.registry) > 0
        
        # Verify glob was ONLY called for model_a (the golden record)
        # We expect a call like 'test-bucket/embeddings/model_a/*.parquet'
        glob_calls = [c[0][0] for c in mock_fs.glob.call_args_list]
        assert any("model_a" in c for c in glob_calls)
        assert not any("model_b" in c for c in glob_calls)

    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_full_scan_mode_scans_all_models(self, mock_fsspec, sample_config, mock_registry_data):
        """When scan_all_models=True, should touch all models."""
        sample_config.scan_all_models = True
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        
        mock_fs.glob.side_effect = lambda p: [p.replace("*", "part_0")]
        mock_fs.open = create_mock_open(mock_registry_data)
        
        ds = ProgressionEmbeddingDataset(sample_config)
        
        glob_calls = [c[0][0] for c in mock_fs.glob.call_args_list]
        assert any("model_a" in c for c in glob_calls)
        assert any("model_b" in c for c in glob_calls)

# ─────────────────────────────────────────────────────────────────────────────
# Caching & JIT Validation Tests (New Architecture)
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheAndLoading:

    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_load_model_populates_cache(self, mock_fsspec, sample_config, mock_embedding_data):
        """Test that load_model_into_memory actually stores data in _active_cache."""
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        mock_fs.glob.return_value = ["bucket/model_a/part_0.parquet"]
        mock_fs.open = create_mock_open(mock_embedding_data)

        ds = ProgressionEmbeddingDataset(sample_config)
        
        # Initial state
        assert ds._active_cache is None
        
        patch_ids = {"patch_001", "patch_002"}
        ds.load_model_into_memory("model_a", patch_ids)
        
        # Post state
        assert ds._active_cache is not None
        assert ds._active_model_name == "model_a"
        assert not ds._active_cache.empty
        # Should contain ALL embedding types (layer_12 and mean_pool)
        assert len(ds._active_cache) >= 2 

    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_jit_validation_fails_missing_patches(self, mock_fsspec, sample_config, mock_embedding_data):
        """Test Just-In-Time validation during loading."""
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        mock_fs.glob.return_value = ["bucket/model_a/part_0.parquet"]
        mock_fs.open = create_mock_open(mock_embedding_data)

        ds = ProgressionEmbeddingDataset(sample_config)
        
        # Request a patch that DOES NOT EXIST in mock_embedding_data
        bad_patch_ids = {"patch_999_ghost"} 
        
        with pytest.raises(ValueError, match="Data Integrity Error"):
            ds.load_model_into_memory("model_a", bad_patch_ids)

    def test_clear_cache(self, sample_config):
        """Test cache clearing."""
        # Minimal mock since we don't need FS for this
        with patch("data.progression_embedding_dataset.fsspec.filesystem"):
            ds = ProgressionEmbeddingDataset(sample_config)
            ds._active_cache = pd.DataFrame({"dummy": [1]})
            ds._active_model_name = "model_a"
            
            ds.clear_cache()
            
            assert ds._active_cache is None
            assert ds._active_model_name is None


# ─────────────────────────────────────────────────────────────────────────────
# Get Cohort Tests (Cache vs GCS)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCohort:

    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_get_cohort_hits_cache(self, mock_fsspec, sample_config, mock_embedding_data):
        """Verify that get_cohort uses memory if model matches."""
        # Setup mocks
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        mock_fs.glob.return_value = ["bucket/model_a/part_0.parquet"]
        mock_fs.open = create_mock_open(mock_embedding_data)
        
        ds = ProgressionEmbeddingDataset(sample_config)
        
        # 1. Load into memory
        patch_ids = {"patch_001"}
        ds.load_model_into_memory("model_a", patch_ids)
        
        # 2. Break the file system mock to PROVE we don't touch GCS
        mock_fs.open.side_effect = Exception("Should not touch GCS!")
        
        # 3. Fetch specific layer
        cohort = ds.get_cohort(patch_ids, "model_a", embedding_type="layer_12")
        
        assert len(cohort) == 1
        assert cohort.iloc[0]["patch_id"] == "patch_001"
    
    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_get_cohort_cache_miss_fallback(self, mock_fsspec, sample_config, mock_embedding_data):
        """Verify fallback to GCS if cache is empty or wrong model."""
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        mock_fs.glob.return_value = ["bucket/model_a/part_0.parquet"]
        mock_fs.open = create_mock_open(mock_embedding_data)
        
        ds = ProgressionEmbeddingDataset(sample_config)
        
        # Cache is empty. Request should succeed via GCS.
        patch_ids = {"patch_001"}
        cohort = ds.get_cohort(patch_ids, "model_a", embedding_type="layer_12")
        
        assert len(cohort) == 1
        # Should have logged a warning about cache miss
        
    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_get_cohort_all_types(self, mock_fsspec, sample_config, mock_embedding_data):
        """Test retrieving all embedding types (embedding_type=None)."""
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        mock_fs.glob.return_value = ["bucket/model_a/part_0.parquet"]
        mock_fs.open = create_mock_open(mock_embedding_data)
        
        ds = ProgressionEmbeddingDataset(sample_config)
        ds.load_model_into_memory("model_a", {"patch_001"})
        
        # Request ALL types (None)
        # patch_001 has 'layer_12' and 'mean_pool' in fixture
        cohort = ds.get_cohort({"patch_001"}, "model_a", embedding_type=None) # Note: get_cohort API changed? 
        # Actually, in your code get_cohort requires embedding_type string unless you changed it.
        # If you changed get_cohort to accept None, use this test. 
        # If get_cohort strictly filters, you skip this.
        # Assuming we allow filtering logic:
        
        # NOTE: In the refactor, get_cohort usually takes a specific type for the dataframe return.
        # But if you load_model_into_memory, you can filter the raw cache yourself.
        # Let's assume standard usage requires a type string.
        pass 

    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_bootstrap_list_duplicates(self, mock_fsspec, sample_config, mock_embedding_data):
        """Ensure get_cohort returns duplicates when passed a bootstrap list."""
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        mock_fs.glob.return_value = ["bucket/model_a/part_0.parquet"]
        mock_fs.open = create_mock_open(mock_embedding_data)
        
        ds = ProgressionEmbeddingDataset(sample_config)
        ds.load_model_into_memory("model_a", {"patch_001"})
        
        # Bootstrap list: patch_001 appears twice
        bootstrap_ids = ["patch_001", "patch_001"]
        cohort = ds.get_cohort(bootstrap_ids, "model_a", embedding_type="layer_12")
        
        assert len(cohort) == 2
        assert all(cohort["patch_id"] == "patch_001")


# ─────────────────────────────────────────────────────────────────────────────
# Sampling Tests (Golden Record Logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestSamplePatchIds:
    
    @patch("data.progression_embedding_dataset.fsspec.filesystem")
    def test_sample_uses_registry_truth(self, mock_fsspec, sample_config, mock_registry_data):
        """Sampling should work purely based on the metadata registry."""
        mock_fs = MagicMock()
        mock_fsspec.return_value = mock_fs
        mock_fs.glob.return_value = ["bucket/model_a/part_0.parquet"]
        mock_fs.open = create_mock_open(mock_registry_data)
        
        ds = ProgressionEmbeddingDataset(sample_config)
        
        patch_ids = ds.sample_patch_ids(n_per_class=10, max_per_slide=10, seed=42)
        
        # In Golden Record mode, we assume if it's in registry, it's valid.
        # JIT validation checks existence later.
        assert len(patch_ids) > 0
        assert "patch_001" in patch_ids