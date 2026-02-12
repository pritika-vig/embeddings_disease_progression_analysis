# Analysis Code

This directory contains the code used to compute diffusion pseudotime (DPT) from previously generated embeddings and to run downstream analyses and plotting.

All scripts should be run from the `analyze_embeddings/` directory.

## Configuration (`config.py`)

Before running any scripts, review `config.py` to ensure it matches your setup. The key configuration sections are:

**`PROGRESSIONS`** — List of disease progressions to evaluate. Each entry specifies a GCS bucket, embedding prefix, ordered class list, root class (earliest stage), and image subdirectory. Uncomment/add entries to control which progressions are included in a run.

**`EXPECTED_MODELS`** — The 6 foundation models to evaluate: `virchow2`, `uni2`, `conch`, `gigapath`, `musk`, `dinov2`.

**`EVALUATION`** — Sampling and evaluation parameters:
- `n_per_class` (default: 1000) — number of patches sampled per disease stage
- `max_per_slide` (default: 50) — cap per slide to reduce slide-level bias
- `n_bootstrap` (default: 10) — bootstrap iterations (some scripts override this)
- `seed` (default: 42) — random seed for reproducibility
- `embedding_type` (default: `"final_embedding"`) — which embedding layer to use

**`DPT`** — Diffusion pseudotime parameters:
- `n_neighbors` (default: 100) — k for the k-NN graph
- `n_diffusion_components` (default: 10) — number of diffusion components

**Embedding type lists** — `PATCH_EMBEDDINGS`, `CLS_EMBEDDINGS`, `REGISTER_EMBEDDINGS`, and `FINAL_EMBEDDING` define which intermediate-layer embeddings to evaluate.

## Output structure

Each data-generating script writes results to a timestamped subdirectory inside `results/`:

```
results/
  2026-02-12_143000_dpt_eval/
    full_manifold_evaluation.csv
  2026-02-12_150000_null_eval/
    null_manifold_evaluation.csv
  latest -> results/2026-02-12_150000_null_eval/   (symlink to most recent run)
```

The `results/latest` symlink is automatically updated to point to whichever script ran most recently. Plotting scripts read through this symlink via config constants (e.g. `config.FULL_RESULTS_OUTPUT_PATH`), so they always pick up the latest data without path changes.

---

## Data generation scripts (`generate_data/`)

### 1. Full manifold evaluation (`evaluate_dpt.py`)

Computes DPT and a suite of manifold quality metrics (Kendall's tau, spectral gap, neighborhood purity, silhouette, trustworthiness, intrinsic dimension, pairwise AUROC) for every combination of progression, model, and embedding type. This is the primary evaluation script. Runs 100 bootstrap iterations on `final_embedding` to produce confidence intervals; other embedding types get a single point estimate.

**Config fields used:**
- `PROGRESSIONS` — which disease progressions to evaluate
- `EXPECTED_MODELS` — which models to evaluate
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters
- `PATCH_EMBEDDINGS`, `CLS_EMBEDDINGS`, `REGISTER_EMBEDDINGS` — layer-wise embedding types to scan

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--n_bootstrap` | 100 | Number of bootstrap iterations for confidence intervals |
| `--output` | (auto) | Override output CSV path |

**Example:**
```bash
# Full run (all progressions, all models, 100 bootstraps) — takes ~12 hours
python generate_data/evaluate_dpt.py

# Quick test with fewer bootstraps
python generate_data/evaluate_dpt.py --n_bootstrap 5
```

**Output:** `results/<timestamp>_dpt_eval/full_manifold_evaluation.csv`

---

### 2. Null distribution (`generate_null_distribution.py`)

Generates a null baseline by computing DPT on real data, then shuffling class labels and recalculating Kendall's tau. This measures what tau you'd expect by chance, providing a statistical baseline for the real results.

**Config fields used:**
- `PROGRESSIONS` — which disease progressions to evaluate
- `EXPECTED_MODELS` — which models to evaluate
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--n_permutations` | 10 | Number of label-shuffle permutations per model |
| `--output` | (auto) | Override output CSV path |

**Example:**
```bash
# Standard run
python generate_data/generate_null_distribution.py

# More permutations for tighter null CI
python generate_data/generate_null_distribution.py --n_permutations 100
```

**Output:** `results/<timestamp>_null_eval/null_manifold_evaluation.csv`

---

### 3. Stage permutation test (`evaluate_stage_permutations.py`)

Exact permutation test that verifies high tau is specific to the biological ordering. Computes tau for ALL possible orderings of disease stages (3 classes = 6 permutations, 4 classes = 24). If the canonical ordering consistently yields the highest tau, the signal is biologically specific.

**Config fields used:**
- `PROGRESSIONS` — which disease progressions to evaluate (uses `classes` to generate permutations)
- `EXPECTED_MODELS` — which models to evaluate
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--output` | (auto) | Override output CSV path |

**Example:**
```bash
python generate_data/evaluate_stage_permutations.py
```

**Output:** `results/<timestamp>_permutation_test/stage_permutation_specificity.csv`

---

### 4. Generalizability analysis (`evaluate_generalizability.py`)

Leave-one-out cross-disease evaluation. For each target progression, calculates the average tau on the other progressions ("reference score"), then trains few-shot logistic regression probes (5, 10, 20 shots per class) on the target. Tests the hypothesis that high-tau models are more data-efficient on unseen diseases.

**Prerequisite:** Requires a completed `full_manifold_evaluation.csv` from script 1 (reads via `config.FULL_RESULTS_OUTPUT_PATH`). Also requires all 4 progressions to be uncommented in `config.PROGRESSIONS`.

**Config fields used:**
- `PROGRESSIONS` — must include all progressions for leave-one-out to work
- `FULL_RESULTS_OUTPUT_PATH` — reads tau values from the main evaluation
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide` — sampling parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--output` | (auto) | Override output CSV path |

**Example:**
```bash
python generate_data/evaluate_generalizability.py
```

**Output:** `results/<timestamp>_generalizability/generalizability_results.csv`

---

### 5. Cell count trajectory analysis (`analyze_trajectory_cellcounts.py`)

Computes DPT for a single progression/model, then runs HistoPLUS cell segmentation on each patch to extract per-cell-type counts along the pseudotime trajectory. Requires local image files and the HistoPLUS package.

**Config fields used:**
- `PROGRESSIONS` — looks up the named progression
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed`, `EVALUATION.embedding_type` — sampling and embedding parameters
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progression` | `CRC-Conventional` | Disease progression name (must match a `config.PROGRESSIONS` entry) |
| `--model` | `uni2` | Foundation model to use for pseudotime |
| `--n_per_class` | 1000 (from config) | Patches per class |
| `--output` | (auto) | Override output CSV path |
| `--test` | off | Test mode: only ~10 patches |
| `--seed` | 42 (from config) | Random seed |
| `--data_dir` | (auto from bucket) | Override local image directory |

**Example:**
```bash
# Quick test
python generate_data/analyze_trajectory_cellcounts.py --test

# Full run for a specific progression and model
python generate_data/analyze_trajectory_cellcounts.py --progression BDC --model uni2 --n_per_class 500

# Custom image directory
python generate_data/analyze_trajectory_cellcounts.py --data_dir ~/data/spider-breast
```

**Output:** `results/<timestamp>_cellcounts/<progression>_<model>_<n>_cellcounts.csv`

---

### 6. Hyperparameter sweep (`hyper_param_sweep.py`)

Sweeps over the number of nearest neighbors (k) used in DPT graph construction, measuring tau, trustworthiness, and spectral gap at each value. Also generates a visualization plot.

**Config fields used:**
- `PROGRESSIONS` — looks up the target progression
- `EXPECTED_MODELS` — which models to sweep
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progression` | `CRC-Serrated` | Target progression |
| `--k_values` | `10 30 50 75 100 200` | Space-separated list of k values to test |
| `--output` | (auto) | Override output CSV path |

**Example:**
```bash
# Default sweep
python generate_data/hyper_param_sweep.py

# Sweep a different progression with custom k values
python generate_data/hyper_param_sweep.py --progression BDC --k_values 25 50 100 150
```

**Output:** `results/<timestamp>_hyperparam_sweep/hyperparam_sweep_k_results.csv` and `k_sweep_analysis.png`

---

## Directory structure

**`generate_data/`**
Data generation scripts (described above). Each writes timestamped output to `results/`.

**`analysis/`**
Core library code. `dpt.py` contains the DPT implementation, metric functions, and data structures (`DPTConfig`, `DPTResult`, `DPTMetric`) used by the generation scripts.

**`data/`**
Dataset loader (`progression_embedding_dataset.py`) that handles GCS-backed embedding retrieval, slide-aware sampling, and in-memory caching.

**`results/`**
All CSV outputs from data generation scripts. Each run creates a timestamped subdirectory; `results/latest` symlinks to the most recent one.

**`plotting/`**
Scripts for generating publication-ready figures. Reads data from `results/latest/` via config path constants.

**`plots/`**
Rendered figures produced by the plotting scripts, organized by analysis type (e.g. `emergence/`, `trajectory_fidelity/`, `permutation/`).
