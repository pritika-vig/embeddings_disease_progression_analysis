import pandas as pd
import numpy as np
import fsspec
import anndata as ad
from typing import List, Dict, Optional
from tqdm.auto import tqdm

class ProgressionEmbeddingDataset:
    """
    Robust loader for pathology foundation model embeddings stored in GCS.
    
    Design:
    - No default parameters allowed (must be supplied by config).
    - Enforces a 'Frozen Cohort' strategy: samples patches ONCE based on a 
      reference model, then joins all other models to this exact list.
    """

    def __init__(self,
                 bucket: str,
                 prefix: str,
                 reference_model: str,
                 ordered_classes: List[str],
                 progression_name: str,
                 patches_per_class: int,
                 per_slide_cap: int,
                 random_state: int,
                 gcs_project: Optional[str] = None):
        
        # 1. Assignment (No Defaults)
        self.bucket = bucket
        self.prefix = prefix
        self.reference_model = reference_model
        self.ordered_classes = ordered_classes
        self.progression_name = progression_name
        self.patches_per_class = patches_per_class
        self.per_slide_cap = per_slide_cap
        self.random_state = random_state
        
        # 2. Infrastructure
        self.fs = fsspec.filesystem('gcs', project=gcs_project)
        self.rng = np.random.default_rng(self.random_state)
        
        self._cohort_metadata: pd.DataFrame = None
        self._embedding_cache = {}

        print(f"🔹 Initializing Cohort: {progression_name}")
        self._build_frozen_cohort()

    def _get_parquet_paths(self, model: str) -> List[str]:
        path = f"gs://{self.bucket}/{self.prefix}/{model}"
        try:
            files = self.fs.glob(f"{path}/*.parquet")
            if not files:
                # Handle edge case where it might be a single file, not a dir
                if self.fs.exists(f"{path}.parquet"):
                    return [f"gs://{path}.parquet"]
                raise FileNotFoundError(f"No parquet files found at {path}")
            return [f"gs://{f}" for f in files]
        except Exception as e:
            raise ValueError(f"Error listing files for model '{model}': {e}")

    def _build_frozen_cohort(self):
        """Scans the reference model to build the master patch list."""
        print(f"   Scanning reference model '{self.reference_model}'...")
        paths = self._get_parquet_paths(self.reference_model)
        
        # Load metadata only
        dfs = []
        for p in tqdm(paths, desc="Loading Metadata", leave=False):
            with self.fs.open(p, 'rb') as f:
                df = pd.read_parquet(f, columns=['dataset', 'class', 'slide_id', 'patch_id'])
                df = df[df['class'].isin(self.ordered_classes)]
                dfs.append(df)
        
        full_meta = pd.concat(dfs, ignore_index=True).drop_duplicates()
        
        # Slide-aware sampling
        sampled_chunks = []
        print(f"   Sampling {self.patches_per_class} patches/class (Cap: {self.per_slide_cap}/slide)...")
        
        for cls_name in self.ordered_classes:
            cls_df = full_meta[full_meta['class'] == cls_name]
            slides = cls_df['slide_id'].unique()
            self.rng.shuffle(slides)
            
            current_count = 0
            for slide in slides:
                if current_count >= self.patches_per_class:
                    break
                slide_patches = cls_df[cls_df['slide_id'] == slide]
                
                # Cap logic
                remaining_needed = self.patches_per_class - current_count
                n_take = min(len(slide_patches), self.per_slide_cap, remaining_needed)
                
                taken = slide_patches.sample(n=n_take, random_state=self.rng.bit_generator)
                sampled_chunks.append(taken)
                current_count += n_take
            
        self._cohort_metadata = pd.concat(sampled_chunks, ignore_index=True)
        
        # Map class names to integer stages (0, 1, 2...) for correlation
        stage_map = {name: i for i, name in enumerate(self.ordered_classes)}
        self._cohort_metadata['stage_int'] = self._cohort_metadata['class'].map(stage_map)
        
        print(f"✅ Cohort Frozen. Total patches: {len(self._cohort_metadata)}")

    def get_embeddings(self, model: str, layer_key: str) -> pd.DataFrame:
        """Fetches embeddings strictly matching the frozen cohort."""
        cache_key = f"{model}_{layer_key}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key].copy()
            
        print(f"   Fetching embeddings for {model} :: {layer_key}...")
        paths = self._get_parquet_paths(model)
        cols = ['class', 'slide_id', 'patch_id', 'embedding_type', 'embedding']
        
        loaded_dfs = []
        for p in tqdm(paths, desc="Reading Shards", leave=False):
            with self.fs.open(p, 'rb') as f:
                shard = pd.read_parquet(f, columns=cols)
                shard = shard[shard['embedding_type'] == layer_key]
                if shard.empty: continue
                
                # Strict Inner Join enforces the Frozen Cohort
                merged = pd.merge(
                    self._cohort_metadata[['class', 'slide_id', 'patch_id', 'stage_int']],
                    shard,
                    on=['class', 'slide_id', 'patch_id'],
                    how='inner'
                )
                loaded_dfs.append(merged)
            
        if not loaded_dfs:
            raise ValueError(f"No embeddings found for {model} (layer: {layer_key})")

        final_df = pd.concat(loaded_dfs, ignore_index=True)
        self._embedding_cache[cache_key] = final_df
        return final_df

    def to_anndata(self, df: pd.DataFrame) -> ad.AnnData:
        """Converts DataFrame to AnnData for Scanpy."""
        # Stack list-of-floats into 2D numpy array
        X = np.stack(df['embedding'].values)
        obs = df.drop(columns=['embedding', 'embedding_type'])
        
        adata = ad.AnnData(X=X, obs=obs)
        
        # Set categorical order for plotting/DPT
        adata.obs['class'] = pd.Categorical(
            adata.obs['class'], 
            categories=self.ordered_classes, 
            ordered=True
        )
        return adata

    # def get_bootstrapped_samples(self, df: pd.DataFrame, n_bootstraps: int) -> List[pd.DataFrame]:
    #     """Performs Cluster Bootstrapping (resampling slides with replacement)."""
    #     bootstraps = []
    #     slides = df['slide_id'].unique()
    #     n_slides = len(slides)
        
    #     for _ in range(n_bootstraps):
    #         # 1. Resample Slides
    #         resampled_slides = self.rng.choice(slides, size=n_slides, replace=True)
            
    #         # 2. Reconstruct Dataset (allow duplicates if slide selected >1 time)
    #         # Efficient list comprehension + single concat
    #         dfs = [df[df['slide_id'] == slide] for slide in resampled_slides]
    #         bootstraps.append(pd.concat(dfs, ignore_index=True))
            
    #     return bootstraps

    def get_bootstrapped_samples(self, df: pd.DataFrame, n_bootstraps: int) -> List[pd.DataFrame]:
        """
        Joint Cluster Bootstrap: Resamples slides globally to preserve 
        cross-class correlations (e.g., Normal & Tumor from same patient).
        """
        bootstraps = []
        # 1. Get unique slides across the ENTIRE dataset (all classes)
        all_slides = df['slide_id'].unique()
        n_slides = len(all_slides)
        
        for _ in range(n_bootstraps):
            # 2. Resample slides with replacement
            resampled_slides = self.rng.choice(all_slides, size=n_slides, replace=True)
            
            # 3. Reconstruct dataset (preserving Class A + Class B coupling)
            # We group by slide first for O(1) lookup speed vs O(N) filtering
            slide_groups = dict(tuple(df.groupby('slide_id')))
            
            parts = []
            for slide in resampled_slides:
                if slide in slide_groups:
                    parts.append(slide_groups[slide])
            
            bootstraps.append(pd.concat(parts, ignore_index=True))
            
        return bootstraps