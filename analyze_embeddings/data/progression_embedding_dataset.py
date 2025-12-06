import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Union

import fsspec
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

@dataclass
class RegistryConfig:
    """Configuration for the ProgressionEmbeddingDataset."""
    bucket: str
    prefix: str
    ordered_classes: List[str]
    models: List[str]
    progression_name: str 
    scan_all_models: bool

class RegistryColumns:
    MODEL = "model"
    CLASS = "class"
    SLIDE_ID = "slide_id"
    PATCH_ID = "patch_id"
    EMBEDDING_TYPE = "embedding_type"
    EMBEDDING = "embedding"

    @classmethod
    def metadata_columns(cls) -> List[str]:
        return [cls.CLASS, cls.SLIDE_ID, cls.PATCH_ID, cls.EMBEDDING_TYPE]

    @classmethod
    def all_columns(cls) -> List[str]:
        return [cls.MODEL] + cls.metadata_columns()

    @classmethod
    def cohort_columns(cls) -> List[str]:
        return [cls.PATCH_ID, cls.CLASS, cls.SLIDE_ID, cls.EMBEDDING]

class ProgressionEmbeddingDataset:
    """
    A Metadata Registry for Pathology Embeddings with In-Memory Caching.

    1. Init: Scans GCS for METADATA only. Quick.
    2. Sample: Filters registry to find common patches.
    3. Load: `load_model_into_memory` brings all layers for a model into RAM.
    4. Fetch: `get_cohort` serves from RAM if loaded, otherwise falls back to GCS.
    """

    def __init__(self, config: RegistryConfig):
        self.config = config
        self.fs = fsspec.filesystem("gcs", project=None)
        
        # Metadata Registry
        self._log_initialization()
        self.registry = self._build_registry()
        self._log_registry_summary()

        # Active Model Cache (Embeddings)
        self._active_cache: Optional[pd.DataFrame] = None
        self._active_model_name: Optional[str] = None

    @property
    def bucket(self) -> str:
        return self.config.bucket

    @property
    def classes(self) -> List[str]:
        return self.config.ordered_classes

    @property
    def models(self) -> List[str]:
        return self.config.models

    # --------------------------------------------------------------------------
    # 1. Sampling & Registry Logic (Metadata Only)
    # --------------------------------------------------------------------------

    def sample_patch_ids(
        self,
        n_per_class: int,
        max_per_slide: int,
        seed: int = 42
    ) -> Set[str]:
        """Slide-aware sampling of patch_ids that exist in ALL models."""
        valid_patches = set(self.registry[RegistryColumns.PATCH_ID].unique())
        rng = np.random.default_rng(seed)

        valid_registry = self.registry[
            self.registry[RegistryColumns.PATCH_ID].isin(valid_patches)
        ].drop_duplicates(subset=[RegistryColumns.PATCH_ID])

        sampled = set()

        for cls in self.classes:
            class_df = valid_registry[valid_registry[RegistryColumns.CLASS] == cls]
            slides = class_df.groupby(RegistryColumns.SLIDE_ID)[RegistryColumns.PATCH_ID].apply(list).to_dict()

            slide_ids = list(slides.keys())
            rng.shuffle(slide_ids)

            class_samples = []
            for slide_id in slide_ids:
                if len(class_samples) >= n_per_class:
                    break

                patches = slides[slide_id]
                rng.shuffle(patches)

                available = n_per_class - len(class_samples)
                take = min(len(patches), max_per_slide, available)
                class_samples.extend(patches[:take])

            sampled.update(class_samples)
            logger.info("Class '%s': sampled %d patches from %d slides", cls, len(class_samples), len(slides))

        return sampled

    def bootstrap_patch_ids(self, patch_ids: Set[str], seed: int) -> List[str]:
        """Stratified bootstrap resample (with replacement) by class."""
        rng = np.random.default_rng(seed)

        patch_classes = self.registry[
            self.registry[RegistryColumns.PATCH_ID].isin(patch_ids)
        ].drop_duplicates(subset=[RegistryColumns.PATCH_ID])[
            [RegistryColumns.PATCH_ID, RegistryColumns.CLASS]
        ]

        bootstrapped = []
        for cls in self.classes:
            class_patches = patch_classes[
                patch_classes[RegistryColumns.CLASS] == cls
            ][RegistryColumns.PATCH_ID].tolist()

            resampled = rng.choice(class_patches, size=len(class_patches), replace=True)
            bootstrapped.extend(resampled.tolist())

        return bootstrapped

    # --------------------------------------------------------------------------
    # 2. Caching & Data Loading (Heavy Lifting)
    # --------------------------------------------------------------------------

    def load_model_into_memory(self, model: str, patch_ids: Set[str], embedding_type: str = None):
        """
        Download ALL embedding types for the specified patches of ONE model.
        Includes Just-In-Time (JIT) validation.
        """
        if self._active_model_name == model and self._active_cache is not None:
            logger.info(f"Model {model} is already loaded in memory.")
            return

        logger.info(f"Pre-loading all embeddings for model: {model}...")
        
        # 1. Fetch data with Error Handling for "Total Failure"
        try:
            df = self._fetch_from_gcs(patch_ids, model, embedding_type=embedding_type)
        except ValueError as e:
            # If _fetch_from_gcs finds 0 matching rows, it raises "No embeddings found".
            # In the context of loading specific patches, this means 100% Data Loss.
            if "No embeddings found" in str(e):
                raise ValueError(
                    f"Data Integrity Error: Model '{model}' is missing all {len(patch_ids)} "
                    f"patches that were present in the Golden Record."
                ) from e
            raise e  # Re-raise unrelated errors

        # 2. Validate "Partial Failure" (missing subsets)
        loaded_ids = set(df[RegistryColumns.PATCH_ID].unique())
        missing = patch_ids - loaded_ids
        
        if missing:
            raise ValueError(
                f"Data Integrity Error: Model '{model}' is missing {len(missing)} "
                f"patches that were present in the Golden Record."
            )
        
        self._active_cache = df
        self._active_model_name = model
        logger.info(f"Verified & Loaded {len(df)} rows into memory for {model}.")

    def clear_cache(self):
        """Free memory."""
        self._active_cache = None
        self._active_model_name = None
        import gc
        gc.collect()

    def get_cohort(
        self,
        patch_ids: Union[Set[str], List[str]],
        model: str,
        embedding_type: str
    ) -> pd.DataFrame:
        """
        Fetch embeddings.
        1. Checks In-Memory Cache first.
        2. Falls back to GCS download if cache is empty or mismatched.
        """
        # A. FAST PATH: Serve from Memory
        if self._active_model_name == model and self._active_cache is not None:
            # Filter the cached dataframe
            mask = self._active_cache[RegistryColumns.EMBEDDING_TYPE] == embedding_type
            
            # If patch_ids is a list (bootstrap), we need to merge/duplicate rows
            # If patch_ids is a set (sampling), we just filter
            subset = self._active_cache[mask]
            
            if isinstance(patch_ids, list):
                # Bootstrap case: Reconstruct based on list order (allows duplicates)
                patch_df = pd.DataFrame({RegistryColumns.PATCH_ID: patch_ids})
                # Left merge preserves the order and duplicates of patch_ids
                merged = patch_df.merge(subset, on=RegistryColumns.PATCH_ID, how="left")
                return merged[RegistryColumns.cohort_columns()]
            
            else:
                # Standard case: Filter by set
                subset = subset[subset[RegistryColumns.PATCH_ID].isin(patch_ids)]
                return subset[RegistryColumns.cohort_columns()]

        # B. SLOW PATH: Go to GCS
        # (This happens if you forget to call load_model_into_memory, or for one-off checks)
        logger.warning(f"⚠️ Cache miss for {model}. Fetching from GCS (Slow).")
        return self._fetch_from_gcs(patch_ids, model, embedding_type)[RegistryColumns.cohort_columns()]

    # --------------------------------------------------------------------------
    # 3. GCS I/O Helpers (Internal)
    # --------------------------------------------------------------------------

    def _fetch_from_gcs(
        self,
        patch_ids: Union[Set[str], List[str]],
        model: str,
        embedding_type: Optional[str]
    ) -> pd.DataFrame:
        """Internal helper to download from GCS."""
        parquet_files = self._find_parquet_files(model)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for model: {model}")

        chunks = []
        for path in parquet_files:
            df = self._read_cohort_parquet(path, patch_ids, embedding_type)
            if df is not None and not df.empty:
                chunks.append(df)

        if not chunks:
            raise ValueError(f"No embeddings found for model={model}")

        result = pd.concat(chunks, ignore_index=True)

        if isinstance(patch_ids, list):
            patch_df = pd.DataFrame({RegistryColumns.PATCH_ID: patch_ids})
            result = patch_df.merge(result, on=RegistryColumns.PATCH_ID, how="left")

        return result

    def _read_cohort_parquet(
        self,
        path: str,
        patch_ids: Union[Set[str], List[str]],
        embedding_type: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Read embeddings. If embedding_type is None, reads ALL types."""
        gcs_path = f"gs://{path}" if not path.startswith("gs://") else path
        patch_set = set(patch_ids)

        try:
            with self.fs.open(gcs_path, "rb") as f:
                # Always read EMBEDDING_TYPE so we can cache/filter it
                cols = RegistryColumns.cohort_columns() + [RegistryColumns.EMBEDDING_TYPE]
                
                df = pd.read_parquet(f, columns=cols)
                
                mask = df[RegistryColumns.PATCH_ID].isin(patch_set)
                if embedding_type is not None:
                    mask = mask & (df[RegistryColumns.EMBEDDING_TYPE] == embedding_type)
                
                return df.loc[mask, cols]
        except Exception as e:
            logger.error("Error reading cohort from %s: %s", path, e)
            return None

    # --------------------------------------------------------------------------
    # 4. Registry Building (Metadata)
    # --------------------------------------------------------------------------
        
    def _build_registry(self) -> pd.DataFrame:
        """
        Builds the registry based on the configuration mode.
        """
        if self.config.scan_all_models:
            return self._build_full_registry_for_validation()
        else:
            return self._build_golden_registry_for_analysis()

    def _build_golden_registry_for_analysis(self) -> pd.DataFrame:
        """FAST: Only scans the first model to get the valid patch list."""
        source_model = self.models[0]
        logger.info(f"Building Index using '{source_model}' as Golden Record...")
        
        parquet_files = self._find_parquet_files(source_model)
        chunks = self._read_parquet_files(parquet_files)
        
        if not chunks:
            # Fallback or error
            return self._empty_registry()

        df = pd.concat(chunks, ignore_index=True)
        # Drop duplicates to keep registry small (one row per patch)
        return df.drop_duplicates(subset=[RegistryColumns.PATCH_ID])

    def _build_full_registry_for_validation(self) -> pd.DataFrame:
        """SLOW: Scans EVERY model to populate 'ds.registry' with all metadata."""
        logger.info("Scanning all models (this may take time)...")
        all_metadata = []

        for model in tqdm(self.models, desc="Indexing All Models"):
            model_df = self._load_model_metadata(model)
            if model_df is not None:
                all_metadata.append(model_df)

        if not all_metadata:
            return self._empty_registry()

        return pd.concat(all_metadata, ignore_index=True)

    def _load_model_metadata(self, model: str) -> Optional[pd.DataFrame]:
        parquet_files = self._find_parquet_files(model)
        if not parquet_files: return None
        chunks = self._read_parquet_files(parquet_files)
        if not chunks: return None
        model_df = pd.concat(chunks, ignore_index=True)
        model_df[RegistryColumns.MODEL] = model
        return model_df

    def _find_parquet_files(self, model: str) -> List[str]:
        base_path = f"{self.config.bucket}/{self.config.prefix}/{model}"
        return self.fs.glob(f"{base_path}/*.parquet")

    def _read_parquet_files(self, file_paths: List[str]) -> List[pd.DataFrame]:
        chunks = []
        for path in file_paths:
            df = self._read_single_parquet(path)
            if df is not None and not df.empty:
                chunks.append(df)
        return chunks

    def _read_single_parquet(self, path: str) -> Optional[pd.DataFrame]:
        gcs_path = f"gs://{path}" if not path.startswith("gs://") else path
        try:
            with self.fs.open(gcs_path, "rb") as f:
                df = pd.read_parquet(f, columns=RegistryColumns.metadata_columns())
                return df[df[RegistryColumns.CLASS].isin(self.classes)]
        except Exception as e:
            logger.error("Error reading %s: %s", path, e)
        return None

    def _empty_registry(self) -> pd.DataFrame:
        return pd.DataFrame(columns=RegistryColumns.all_columns())

    def _log_initialization(self) -> None:
        logger.info("Building Metadata Registry: %s", self.config.progression_name)
        logger.info("Classes: %s", self.classes)
        logger.info("Models: %s", self.models)

    def _log_registry_summary(self) -> None:
        logger.info("Registry Built. Total Rows: %d", len(self.registry))