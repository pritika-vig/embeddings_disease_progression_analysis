"""
Diffusion Pseudotime (DPT) Analysis Utilities.

This module provides tools to construct diffusion maps from pathology embeddings,
calculate pseudotime trajectories, and evaluate manifold quality using various
metrics (Kendall's Tau, Intrinsic Dimension, Trustworthiness, etc.).
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

# Handle optional dependency for Intrinsic Dimension
try:
    import skdim
    HAS_SKDIM = True
except ImportError:
    HAS_SKDIM = False

# Suppress verbose Scanpy/AnnData warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration & Data Structures
# =============================================================================

class DPTMetric(Enum):
    """Available metrics for manifold evaluation."""
    TAU = "tau"
    P_VALUE = "p_value"
    SPECTRAL_GAP = "spectral_gap"
    NEIGHBORHOOD_PURITY = "neighborhood_purity"
    TRUSTWORTHINESS = "trustworthiness"
    ID_RAW = "id_raw"       # Intrinsic Dimension of original embeddings
    ID_DIFF = "id_diff"     # Intrinsic Dimension of diffusion components


@dataclass
class DPTConfig:
    """
    Configuration for Diffusion Pseudotime analysis.
    
    Attributes:
        n_neighbors: Number of nearest neighbors for graph construction.
        n_diffusion_components: Number of diffusion components to calculate.
        metrics: Set of metrics to compute. Defaults to ALL metrics.
        subsample_size: Number of points to use for expensive metrics (ID/Trust).
    """
    n_neighbors: int
    n_diffusion_components: int
    metrics: Set[DPTMetric] = field(default_factory=lambda: set(DPTMetric))
    subsample_size: int = 2000


@dataclass
class DPTResult:
    """
    Results container for DPT analysis.
    Fields default to np.nan if the metric was not requested or computation failed.
    """
    tau: float = np.nan
    p_value: float = np.nan
    spectral_gap: float = np.nan
    neighborhood_purity: float = np.nan
    trustworthiness: float = np.nan
    id_raw: float = np.nan
    id_diff: float = np.nan

    @property
    def is_valid(self) -> bool:
        """Check if the primary metric (Tau) was successfully computed."""
        return not np.isnan(self.tau)

    def to_dict(self) -> dict:
        """Convert result to a dictionary."""
        return {
            "tau": self.tau,
            "p_value": self.p_value,
            "spectral_gap": self.spectral_gap,
            "neighborhood_purity": self.neighborhood_purity,
            "trustworthiness": self.trustworthiness,
            "id_raw": self.id_raw,
            "id_diff": self.id_diff,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def cohort_to_anndata(cohort_df: pd.DataFrame, ordered_classes: List[str]) -> sc.AnnData:
    """
    Convert a cohort DataFrame into a Scanpy AnnData object.
    
    Args:
        cohort_df: DataFrame with 'embedding', 'patch_id', 'class', 'slide_id'.
        ordered_classes: List of class names in progression order.
    """
    X = np.vstack(cohort_df["embedding"].values)
    obs = cohort_df[["patch_id", "class", "slide_id"]].copy()
    
    # Map string classes to integer stages
    class_map = {c: i for i, c in enumerate(ordered_classes)}
    obs["stage_int"] = obs["class"].map(class_map)
    
    return sc.AnnData(X=X, obs=obs)


def find_medoid(adata: sc.AnnData, class_name: str) -> Optional[int]:
    """
    Find the index of the medoid (geometric center) for a specific class.
    
    Returns:
        Index of the medoid patch, or None if class is missing.
    """
    mask = adata.obs["class"] == class_name
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return None
    
    X_class = adata.X[mask]
    # Compute centroid
    centroid = X_class.mean(axis=0).reshape(1, -1)
    # Find point closest to centroid (Cosine distance for ViT embeddings)
    distances = cdist(X_class, centroid, metric="cosine")
    
    return indices[np.argmin(distances)]


def _compute_spectral_gap(adata: sc.AnnData) -> float:
    """Calculate eigengap between 1st and 2nd non-trivial diffusion components."""
    if "diffmap_evals" not in adata.uns:
        return np.nan
    evals = adata.uns["diffmap_evals"]
    # evals[0] is always 1.0 (steady state)
    # Gap of interest is usually between DC1 (evals[1]) and DC2 (evals[2])
    if len(evals) > 2:
        return evals[1] - evals[2]
    return np.nan


def _compute_neighborhood_purity(adata: sc.AnnData, label_col: str = "stage_int") -> float:
    """
    Calculate average neighborhood purity.
    Measures local consistency: "Do my neighbors share my label?"
    """
    if "neighbors" not in adata.uns:
        return np.nan
        
    labels = adata.obs[label_col].values
    # Get connectivity matrix (sparse)
    connectivities = adata.obsp["connectivities"]
    
    n_cells = adata.n_obs
    total_purity = 0.0
    
    # Iterate through cells (vectorized neighbor lookup is harder with sparse matrix)
    for i in range(n_cells):
        neighbor_indices = connectivities[i].nonzero()[1]
        if len(neighbor_indices) == 0:
            continue
            
        center_label = labels[i]
        neighbor_labels = labels[neighbor_indices]
        matches = np.sum(neighbor_labels == center_label)
        total_purity += matches / len(neighbor_indices)
        
    return total_purity / n_cells


def _estimate_id(X: np.ndarray) -> float:
    """
    Estimate Intrinsic Dimension (ID).
    Uses 'scikit-dimension' TwoNN if available, else a simple MLE fallback.
    """
    if not np.all(np.isfinite(X)):
        return np.nan
        
    if HAS_SKDIM:
        try:
            dve = skdim.id.TwoNN()
            return dve.fit(X).dimension_
        except Exception:
            return np.nan
    
    # Fallback MLE (if skdim not installed)
    try:
        k = 10
        if X.shape[0] <= k:
            return np.nan
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(X)
        dist, _ = nbrs.kneighbors(X)
        dist = dist[:, 1:] # Drop self
        
        # Simplified MLE
        v = np.log(dist[:, k-1:k] / dist[:, 0:k-1])
        mle = (k - 2) / np.mean(np.sum(v, axis=1))
        return mle
    except Exception:
        return np.nan


def _compute_trustworthiness(adata: sc.AnnData, subsample_size: int) -> float:
    """
    Compute trustworthiness of the embedding.
    Uses subsampling to avoid O(N^2) cost on large datasets.
    """
    if "X_diffmap" not in adata.obsm:
        return np.nan
        
    try:
        n_total = adata.n_obs
        if n_total > subsample_size:
            indices = np.random.choice(n_total, subsample_size, replace=False)
            X_high = adata.X[indices]
            X_low = adata.obsm["X_diffmap"][indices]
        else:
            X_high = adata.X
            X_low = adata.obsm["X_diffmap"]

        return trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    except Exception:
        return np.nan


# =============================================================================
# Main Computation
# =============================================================================

def compute_dpt(adata: sc.AnnData, root_class: str, config: DPTConfig) -> DPTResult:
    """
    Run Diffusion Pseudotime analysis and compute requested metrics.
    
    Args:
        adata: AnnData object containing embeddings.
        root_class: Name of the class to use as the start of the trajectory.
        config: DPTConfig object specifying parameters and metrics.
        
    Returns:
        DPTResult object containing computed metrics.
    """
    result = DPTResult()
    
    try:
        # 1. Construct Neighborhood Graph
        # Use Cosine metric for ViT embeddings; method='gauss' matches DPT paper logic
        sc.pp.neighbors(
            adata,
            use_rep="X",
            n_neighbors=config.n_neighbors,
            metric="cosine",
            method="gauss",
        )
        
        # 2. Compute Diffusion Map
        # Note: alpha=0 is used here for robustness with balanced cohorts
        sc.tl.diffmap(adata)
        
        # 3. Identify Root Cell
        adata.uns['iroot'] = find_medoid(adata, root_class)
        if adata.uns['iroot'] is None:
            logger.warning(f"Root class '{root_class}' medoid not found.")
            return result

        # 4. Compute Diffusion Pseudotime (DPT)
        sc.tl.dpt(adata, n_dcs=config.n_diffusion_components)
        
        # Validate DPT output
        if "dpt_pseudotime" not in adata.obs:
            return result
        
        valid_mask = np.isfinite(adata.obs["dpt_pseudotime"])
        if valid_mask.sum() < 10:
            logger.warning("Insufficient valid pseudotime values computed.")
            return result

        # ==========================
        # Metric Computations
        # ==========================
        
        # 1. Kendall's Tau (Primary Metric)
        if DPTMetric.TAU in config.metrics:
            pt = adata.obs["dpt_pseudotime"][valid_mask]
            stages = adata.obs["stage_int"][valid_mask]
            tau, p_val = kendalltau(pt, stages)
            result.tau = tau
            result.p_value = p_val

        # 2. Spectral Gap (Metastability)
        if DPTMetric.SPECTRAL_GAP in config.metrics:
            result.spectral_gap = _compute_spectral_gap(adata)

        # 3. Neighborhood Purity (Local Consistency)
        if DPTMetric.NEIGHBORHOOD_PURITY in config.metrics:
            result.neighborhood_purity = _compute_neighborhood_purity(adata)

        # 4. Trustworthiness (Embedding Distortion) - Expensive
        if DPTMetric.TRUSTWORTHINESS in config.metrics:
            result.trustworthiness = _compute_trustworthiness(adata, config.subsample_size)

        # 5. Intrinsic Dimension (Raw)
        if DPTMetric.ID_RAW in config.metrics:
            idx = np.random.choice(adata.n_obs, min(adata.n_obs, config.subsample_size), replace=False)
            result.id_raw = _estimate_id(adata.X[idx])

        # 6. Intrinsic Dimension (Diffusion Manifold)
        if DPTMetric.ID_DIFF in config.metrics:
            if "X_diffmap" in adata.obsm:
                idx = np.random.choice(adata.n_obs, min(adata.n_obs, config.subsample_size), replace=False)
                result.id_diff = _estimate_id(adata.obsm['X_diffmap'][idx])

        return result

    except Exception as e:
        logger.debug(f"DPT Computation failed: {e}", exc_info=True)
        return result