# Embedding Generation Code

This directory contains the code used to generate patch-level embeddings from the SPIDER datasets before running any pseudotime or downstream analysis.

## Dataset setup

This code assumes you have manually downloaded and unpacked the following SPIDER datasets into ImageNet-style directory layouts, following the dataset instructions:

- https://huggingface.co/datasets/histai/SPIDER-skin  
- https://huggingface.co/datasets/histai/SPIDER-colorectal  
- https://huggingface.co/datasets/histai/SPIDER-breast  

After unpacking, update the dataset paths in `config.py` so the loader can locate your local directories.

## Output configuration

Embeddings are written incrementally as Parquet files to the Google Cloud Storage bucket specified in `config.py`.  
A metadata file describing the run (model, timestamp, dataset, patching configuration, etc.) is also written to the same destination.

## Running embedding generation

To generate embeddings for all datasets and all models configured in `config.py`:

python main.py

The script will iterate over patches, stream batches through the selected model, and write Parquet shards as it progresses.

## Directory structure

**core/**  
Contains core abstractions for preprocessing, patch streaming, batching, and general embedding-pipeline utilities.

**models/**  
Defines the model interface used by the pipeline, with one implementation per foundation model evaluated in the study.

**config.py**  
Holds all dataset paths, model selection, output bucket configuration, and performance parameters.


