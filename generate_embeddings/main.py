#!/usr/bin/env python3
import argparse
import os
import torch
from pathlib import Path

from config import DATASETS
from core.validator import DryRunValidator
from core.processor import LayerEmbeddingProcessor

def load_hf_token():
    """Load HuggingFace token from environment or file."""
    token = os.getenv("HF_TOKEN")
    if not token and Path("~/.hf_token").expanduser().exists():
        token = Path("~/.hf_token").expanduser().read_text().strip()
    if token:
        os.environ["HF_TOKEN"] = token
        print("✓ HF Token loaded")
    else:
        print("⚠️ No HF Token found. Some models may fail.")

def main():
    parser = argparse.ArgumentParser(description="SPIDER Pathology Embeddings Extractor")
    
    # Dataset Selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str, choices=[d.dataset_name for d in DATASETS],
                       help="Process a specific dataset")
    group.add_argument('--all-datasets', action='store_true',
                       help="Process all datasets sequentially")

    parser.add_argument('--models', nargs='+', 
                       default=['dinov2', 'uni2', 'gigapath', 'conch', 'musk'],
                       help='List of models to process')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate models and GCS access without processing')
    parser.add_argument('--test-run', action='store_true',
                       help='Process only 10 images per class for testing')
    
    args = parser.parse_args()
    load_hf_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine target datasets
    if args.all_datasets:
        targets = DATASETS
    else:
        targets = [d for d in DATASETS if d.dataset_name == args.dataset]

    # Mode Selection
    if args.dry_run:
        print("\n🔎 STARTING DRY RUN VALIDATION")
        val = DryRunValidator(device)
        all_passed = True
        
        for ds in targets:
            if not val.validate_dataset(ds): all_passed = False
            if not val.validate_gcs(ds.bucket): all_passed = False
            
        for m in args.models:
            if not val.validate_model(m): all_passed = False
            
        if all_passed:
            print("\n✅ All validations passed.")
        else:
            print("\n❌ Validation failed. See errors above.")
        return

    # Processing Loop
    processor = LayerEmbeddingProcessor(device)
    for idx, ds in enumerate(targets):
        print(f"\n{'='*60}")
        print(f"Processing Dataset {idx+1}/{len(targets)}: {ds.dataset_name}")
        print(f"{'='*60}")
        processor.process_dataset(ds, args.models, test_run=args.test_run)

if __name__ == "__main__":
    main()