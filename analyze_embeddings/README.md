# Analysis Code

This directory contains the code used to compute diffusion pseudotime (DPT) from previously generated embeddings and to run downstream analyses and plotting.

## Generating diffusion pseudotime

Before running any scripts, confirm that all dataset and output paths in `config.py` point to the embeddings you produced using the generation code.

To compute DPT for all disease progressions:

python evaluate_dpt.py

This script produces a full pseudotime CSV for each progression and stores all outputs in `results/`. Computing the complete set takes about twelve hours.

## Directory structure

**analysis/**  
Contains the main DPT implementation (`dpt.py`) along with supporting scripts for computing additional metrics and summary statistics.

**results/**  
Stores all intermediate and final outputs from pseudotime computation and analysis.

**plotting/**  
Contains code for generating plots, figures, and tables.

**plots/**  
Holds the rendered figures produced by the plotting scripts.
