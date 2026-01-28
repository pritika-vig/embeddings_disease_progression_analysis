# Disease Progression Embedding and Analysis

This repository contains code for generating patch-level embeddings from SPIDER datasets and analyzing them using diffusion pseudotime (DPT). The workflow is split into two main components:

1. `generate/` — generates embeddings using pathology foundation models.  
2. `analyze/` — computes diffusion pseudotime trajectories and downstream metrics from the generated embeddings.

## Repository structure

**generate/**  
Embedding pipeline for SPIDER-skin, SPIDER-colorectal, and SPIDER-breast datasets.  
Includes dataset loaders, model interfaces, and logic for writing embeddings as Parquet shards to a Google Cloud bucket.

**analyze/**  
Runs diffusion pseudotime, computes trajectory fidelity, generates metrics, and produces plots and tables.

**config.py**  
Shared configuration file for dataset paths, model choices, and output locations.

**requirements.txt**  
Pinned Python dependencies generated from the development environment.

## Setup

1. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

2. Install dependencies:


3. Download SPIDER skin, colorectal, and breast datasets and unpack them into ImageNet-style layouts:

- https://huggingface.co/datasets/histai/SPIDER-skin  
- https://huggingface.co/datasets/histai/SPIDER-colorectal  
- https://huggingface.co/datasets/histai/SPIDER-breast  

4. First generate embeddings using the readme instructions in the generate folder, and then analyze embeddings from the analyze folder. 


This computes DPT for each progression and writes outputs to `results/`.  
Additional scripts for metrics and plotting are in `analysis/` and `plotting/`.

## Outputs

- Incremental Parquet embedding shards (Google Cloud bucket)
- Metadata describing each embedding run
- Pseudotime CSVs
- Derived metrics
- Figures and tables

## Notes

- The analysis pipeline expects embeddings produced by the `generate/` step.  
- Pseudotime computation is slow for full cohorts (~12 hours).  
- Slide-aware sampling is implemented in the analysis code to reduce slide-level bias.
