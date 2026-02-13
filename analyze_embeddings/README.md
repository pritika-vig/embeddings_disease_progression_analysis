# Analysis Code

This directory contains the code used to compute diffusion pseudotime (DPT) from previously generated embeddings and to run downstream analyses and plotting.

All scripts should be run from the `analyze_embeddings/` directory.

## Configuration (`config.py`)

Before running any scripts, review `config.py` to ensure it matches your setup. The key configuration sections are:

**`PROGRESSIONS`** — List of all disease progressions. Each entry specifies a GCS bucket, embedding prefix, ordered class list, root class (earliest stage), and image subdirectory. All 4 progressions are always defined; use the `--progressions` CLI argument on each script to select which ones to include in a run (defaults to all).

**`EXPECTED_MODELS`** — The 6 foundation models to evaluate: `virchow2`, `uni2`, `conch`, `gigapath`, `musk`, `dinov2`.

**`EVALUATION`** — Sampling and evaluation parameters:
- `n_per_class` (default: 1000) — number of patches sampled per disease stage
- `max_per_slide` (default: 50) — cap per slide to reduce slide-level bias
- `seed` (default: 42) — random seed for reproducibility
- `embedding_type` (default: `"final_embedding"`) — which embedding layer to use
- `n_bootstrap_ci` (default: 100) — bootstrap resampling iterations for metric confidence intervals (`evaluate_dpt.py`)
- `n_null_permutations` (default: 10) — label-shuffle permutations for null tau distribution (`generate_null_distribution.py`)
- `n_fewshot_trials` (default: 10) — repeated few-shot sampling trials per shot count (`evaluate_generalizability.py`)
- `fewshot_n_shots` (default: `[5, 10, 20]`) — shots per class for few-shot probing (`evaluate_generalizability.py`)

**`TEST_EVALUATION`** — Same keys as `EVALUATION` with minimal values for quick smoke tests (`n_per_class=50`, `n_bootstrap_ci=2`, etc.). Used automatically when any script is run with `--test`.

**`DPT`** — Diffusion pseudotime parameters:
- `n_neighbors` (default: 100) — k for the k-NN graph
- `n_diffusion_components` (default: 10) — number of diffusion components

**Embedding type lists** — `PATCH_EMBEDDINGS`, `CLS_EMBEDDINGS`, `REGISTER_EMBEDDINGS`, and `FINAL_EMBEDDING` define which intermediate-layer embeddings to evaluate.

## Output structure

All scripts write results into a shared date-stamped directory inside `results/`:

```
results/
  2026-02-12/
    full_manifold_evaluation.csv
    null_manifold_evaluation.csv
    stage_permutation_specificity.csv
    generalizability_results.csv
    ...
  2026-02-13/
    full_manifold_evaluation.csv
    ...
  latest -> results/2026-02-13/   (symlink to the most recent day's folder)
  test/                            (--test mode output, not symlinked)
```

All scripts running on the same day write into the same `YYYY-MM-DD/` folder, so `results/latest/` always contains a complete set of outputs. Re-running a script on the same day overwrites its CSV. Plotting scripts read through the `latest` symlink via config constants (e.g. `config.FULL_RESULTS_OUTPUT_PATH`), so they always pick up the latest data without path changes.

The `results/test/` directory is used by `--test` mode (see below) and is never symlinked or read by plotting scripts.

## Test mode

All scripts support a `--test` flag that runs with minimal samples for quick verification that data paths, progressions, and dependencies are working. Test mode:

- Uses `n_per_class=50`, `max_per_slide=10` (instead of 1000/50)
- Reduces bootstrap/permutation iterations to 2
- Writes output to `results/test/` (fixed directory, no timestamp, no symlink update)
- Does **not** affect plotting scripts or `results/latest`

```bash
# Verify a single progression works end-to-end
python generate_data/evaluate_dpt.py --test --progressions BDC

# Verify all progressions
python generate_data/evaluate_dpt.py --test
```

## Running everything

`run_all.py` runs all 6 data generation scripts in sequence with the correct ordering (evaluate_dpt first, since evaluate_generalizability depends on its output). Scripts 1-4 run on all progressions with full config parameters. The hyperparameter sweep runs on its default progression (CRC-Serrated) and trajectory cell counts runs on CRC-Conventional with uni2.

```bash
# Full pipeline
python generate_data/run_all.py

# Smoke test all scripts
python generate_data/run_all.py --test
```

If a script fails, the runner logs the error and continues to the next one. A summary is printed at the end showing pass/fail status and elapsed time for each step.

---

## Data generation scripts (`generate_data/`)

### 1. Full manifold evaluation (`evaluate_dpt.py`)

Computes DPT and a suite of manifold quality metrics (Kendall's tau, spectral gap, neighborhood purity, silhouette, trustworthiness, intrinsic dimension, pairwise AUROC) for every combination of progression, model, and embedding type. This is the primary evaluation script. Runs `n_bootstrap_ci` iterations (default: 100) on `final_embedding` to produce confidence intervals; other embedding types get a single point estimate.

**Config fields used:**
- `PROGRESSIONS` — which disease progressions to evaluate
- `EXPECTED_MODELS` — which models to evaluate
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters
- `EVALUATION.n_bootstrap_ci` — bootstrap iterations for confidence intervals
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters
- `PATCH_EMBEDDINGS`, `CLS_EMBEDDINGS`, `REGISTER_EMBEDDINGS` — layer-wise embedding types to scan

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progressions` | all | Progression names to include (e.g. `--progressions BDC SCC`) |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, final_embedding only, output to `results/test/` |

**Example:**
```bash
# Full run (all progressions, all models) — takes ~12 hours
python generate_data/evaluate_dpt.py

# Run only BDC and SCC progressions
python generate_data/evaluate_dpt.py --progressions BDC SCC

# Quick smoke test
python generate_data/evaluate_dpt.py --test --progressions BDC
```

**Output:** `results/YYYY-MM-DD/full_manifold_evaluation.csv`

---

### 2. Null distribution (`generate_null_distribution.py`)

Generates a null baseline by computing DPT on real data, then shuffling class labels and recalculating Kendall's tau. This measures what tau you'd expect by chance, providing a statistical baseline for the real results.

**Config fields used:**
- `PROGRESSIONS` — which disease progressions to evaluate
- `EXPECTED_MODELS` — which models to evaluate
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters
- `EVALUATION.n_null_permutations` — number of label-shuffle permutations
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progressions` | all | Progression names to include (e.g. `--progressions BDC SCC`) |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, output to `results/test/` |

**Example:**
```bash
# Standard run
python generate_data/generate_null_distribution.py

# Quick smoke test on one progression
python generate_data/generate_null_distribution.py --test --progressions BDC
```

**Output:** `results/YYYY-MM-DD/null_manifold_evaluation.csv`

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
| `--progressions` | all | Progression names to include (e.g. `--progressions BDC SCC`) |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, output to `results/test/` |

**Example:**
```bash
# Full run
python generate_data/evaluate_stage_permutations.py

# Run on specific progressions
python generate_data/evaluate_stage_permutations.py --progressions CRC-Conventional CRC-Serrated

# Quick smoke test
python generate_data/evaluate_stage_permutations.py --test --progressions BDC
```

**Output:** `results/YYYY-MM-DD/stage_permutation_specificity.csv`

---

### 4. Generalizability analysis (`evaluate_generalizability.py`)

Leave-one-out cross-disease evaluation. For each target progression, calculates the average tau on the other progressions ("reference score"), then trains few-shot logistic regression probes (5, 10, 20 shots per class) on the target. Tests the hypothesis that high-tau models are more data-efficient on unseen diseases.

**Prerequisite:** Requires a completed `full_manifold_evaluation.csv` from script 1 (reads via `config.FULL_RESULTS_OUTPUT_PATH`).

**Config fields used:**
- `PROGRESSIONS` — progressions for leave-one-out (default: all 4)
- `FULL_RESULTS_OUTPUT_PATH` — reads tau values from the main evaluation
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide` — sampling parameters
- `EVALUATION.n_fewshot_trials` — repeated sampling trials per shot count
- `EVALUATION.fewshot_n_shots` — shots per class for few-shot probing

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progressions` | all | Progression names to include (e.g. `--progressions BDC SCC CRC-Conventional`) |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, output to `results/test/` |

**Example:**
```bash
# Full run (requires all 4 progressions in full_manifold_evaluation.csv)
python generate_data/evaluate_generalizability.py

# Quick smoke test
python generate_data/evaluate_generalizability.py --test
```

**Output:** `results/YYYY-MM-DD/generalizability_results.csv`

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
| `--test` | off | Test mode: 3 patches per class, output to `results/test/` |
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

**Output:** `results/YYYY-MM-DD/<progression>_<model>_<n>_cellcounts.csv`

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
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, k_values=[50, 100], output to `results/test/` |

**Example:**
```bash
# Default sweep
python generate_data/hyper_param_sweep.py

# Sweep a different progression with custom k values
python generate_data/hyper_param_sweep.py --progression BDC --k_values 25 50 100 150

# Quick smoke test
python generate_data/hyper_param_sweep.py --test
```

**Output:** `results/YYYY-MM-DD/hyperparam_sweep_k_results.csv` and `k_sweep_analysis.png`

---

## Directory structure

**`generate_data/`**
Data generation scripts (described above). Each writes output to `results/YYYY-MM-DD/`.

**`analysis/`**
Core library code. `dpt.py` contains the DPT implementation, metric functions, and data structures (`DPTConfig`, `DPTResult`, `DPTMetric`) used by the generation scripts.

**`data/`**
Dataset loader (`progression_embedding_dataset.py`) that handles GCS-backed embedding retrieval, slide-aware sampling, and in-memory caching.

**`results/`**
All CSV outputs from data generation scripts. Each day's runs share a `YYYY-MM-DD/` subdirectory; `results/latest` symlinks to the most recent one. `results/test/` holds test-mode output.

**`plotting/`**
Scripts for generating publication-ready figures. Reads data from `results/latest/` via config path constants.

**`plots/`**
Rendered figures produced by the plotting scripts, organized by analysis type (e.g. `emergence/`, `trajectory_fidelity/`, `permutation/`).
