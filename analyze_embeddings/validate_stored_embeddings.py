import pandas as pd
from tabulate import tabulate
from data.progression_embedding_dataset import ProgressionEmbeddingDataset, RegistryConfig
from config import PROGRESSIONS, EXPECTED_MODELS

def validate_consistency():
    """
    Validates that for every Class in a Progression, 
    every Model and Layer has the EXACT same patch count.
    """
    
    for prog_config in PROGRESSIONS:
        print(f"\n" + "="*80)
        print(f"🏥 Validating Progression: {prog_config['name']}")
        print("="*80)

        # 1. Load Registry (This scans all Parquet metadata from GCS)
        try:
            registry_config = RegistryConfig(
                bucket=prog_config['bucket'],
                prefix=prog_config['prefix'],
                ordered_classes=prog_config['classes'],
                models=EXPECTED_MODELS,
                progression_name=prog_config['name'],
                scan_all_models = True
            )
            ds = ProgressionEmbeddingDataset(registry_config)
        except Exception as e:
            print(f"Critical Error initializing dataset: {e}")
            continue

        if ds.registry.empty:
            print("Registry is empty. Check GCS paths/permissions.")
            continue

        # 2. Group by [Class, Model, Layer] and Count
        # This creates a table like: | Epidermis | virchow2 | cls_token | 1000 |
        counts = ds.registry.groupby(
            ['class', 'model', 'embedding_type']
        ).size().reset_index(name='count')

        # 3. Check Consistency Per Class
        # We iterate through each class and ensure there is only ONE unique count 
        # across all models and layers.
        
        report_data = []
        headers = ["Class", "Unique Counts Found", "Status", "Details"]

        for cls in prog_config['classes']:
            cls_data = counts[counts['class'] == cls]
            
            if cls_data.empty:
                report_data.append([cls, "0", "❌ MISSING", "No data found for class"])
                continue

            # Get the unique counts found for this class
            unique_counts = cls_data['count'].unique()
            
            if len(unique_counts) == 1:
                status = "✅ PASS"
                count_str = str(unique_counts[0])
                details = f"All {len(cls_data)} model/layers have {count_str} patches"
            else:
                status = "❌ FAIL"
                count_str = str(sorted(unique_counts))
                
                # Find the deviants for the details column
                mode_count = cls_data['count'].mode()[0]
                deviants = cls_data[cls_data['count'] != mode_count]
                deviant_desc = ", ".join([f"{r['model']}/{r['embedding_type']}({r['count']})" for _, r in deviants.iterrows()])
                details = f"Mismatch! Majority={mode_count}. Deviants: {deviant_desc}"

            report_data.append([cls, count_str, status, details])

        # 4. Print Report
        print("\n" + tabulate(report_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    validate_consistency()