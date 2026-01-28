import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
import gc
import json
from tqdm import tqdm
from PIL import Image
from google.cloud import storage
from datetime import datetime

from config import DATA_ROOT, SHARD_SIZE, LAYER_FRACTIONS, OutputConfig
from models import get_model_adapter
from .extractor import LayerExtractor

class LayerEmbeddingProcessor:
    def __init__(self, device):
        self.device = device
        self.client = storage.Client()

    def get_image_list(self, dataset, test_run=False):
        """Scan directory for images."""
        images = []
        root = DATA_ROOT / dataset.dataset_name / "train"
        
        for cls_name in dataset.classes:
            cls_dir = root / cls_name
            if not cls_dir.exists(): continue
            
            count = 0
            for slide_dir in sorted(cls_dir.iterdir()):
                if not slide_dir.is_dir(): continue
                slide_id = slide_dir.name.replace("slide_", "")
                
                for patch_path in sorted(slide_dir.glob("*.png")):
                    images.append({
                        'path': str(patch_path),
                        'class': cls_name,
                        'slide_id': slide_id,
                        'patch_id': patch_path.stem.replace("patch_", "")
                    })
                    count += 1
                    if test_run and count >= 10: break
                if test_run and count >= 10: break
        return images

    def process_dataset(self, dataset, models, test_run=False):
        image_info = self.get_image_list(dataset, test_run)
        print(f"   Found {len(image_info)} images total.")
        
        runtime_meta = {}
        
        for model_name in models:
            print(f"\n   🤖 Processing Model: {model_name}")
            try:
                # Initialize Adapter
                adapter = get_model_adapter(model_name, self.device)
                adapter.load()
                transform = adapter.get_transform()
                
                # Initialize Extractor
                extractor = LayerExtractor(adapter, LAYER_FRACTIONS)
                
                # Save metadata
                model_meta = extractor.get_provenance_meta()
                model_meta['embedding_dim'] = adapter.embed_dim
                runtime_meta[model_name] = model_meta

                # Processing Loop
                batch_size = adapter.default_batch_size
                
                rows = []
                shard_counter = 0
                
                for i in tqdm(range(0, len(image_info), batch_size), desc=f"      {model_name}"):
                    batch_meta = image_info[i:i+batch_size]
                    
                    tensors = []
                    valid_meta = []
                    for meta in batch_meta:
                        try:
                            img = Image.open(meta['path']).convert('RGB')
                            tensors.append(transform(img))
                            valid_meta.append(meta)
                        except Exception as e:
                            print(f"Error loading {meta['path']}: {e}")
                    
                    if not tensors: continue
                    
                    batch_t = torch.stack(tensors).to(self.device)
                    embeddings = extractor.extract(batch_t)
                    
                    # Flatten results
                    for idx, meta in enumerate(valid_meta):
                        for type_key, emb_val in embeddings.items():
                            rows.append({
                                'dataset': dataset.dataset_name,
                                'class': meta['class'],
                                'slide_id': meta['slide_id'],
                                'patch_id': meta['patch_id'],
                                'model': model_name,
                                'embedding_type': type_key,
                                'embedding': emb_val[idx].tolist(),
                                'dim': emb_val[idx].shape[0]
                            })
                    
                    del batch_t

                    # If buffer exceeds SHARD_SIZE, write immediately
                    if len(rows) >= SHARD_SIZE:
                        self.upload_shard(rows, dataset.bucket, model_name, test_run, shard_counter)
                        shard_counter += 1
                        rows = [] # Clear memory
                        gc.collect() # Force garbage collection

                # Upload remaining rows after loop finishes
                if rows:
                    self.upload_shard(rows, dataset.bucket, model_name, test_run, shard_counter)

                # Cleanup Model
                extractor.cleanup()
                adapter.cleanup()
                del extractor, adapter
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ Failed to process {model_name}: {e}")
                import traceback
                traceback.print_exc()

        self.write_provenance(dataset, runtime_meta, test_run)

    def upload_shard(self, rows, bucket_name, model_name, test_run, shard_index):
        """Helper to upload a single shard immediately."""
        df = pd.DataFrame(rows)
        bucket = self.client.bucket(bucket_name)
        prefix = OutputConfig.get_path(test_run)
        
        # Convert to Parquet in memory
        table = pa.Table.from_pandas(df)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression='snappy')
        
        # Upload
        blob_path = f"{prefix}/{model_name}/shard_{shard_index:05d}.parquet"
        bucket.blob(blob_path).upload_from_string(buf.getvalue())
        
        # Explicit delete to help GC
        del df, table, buf

    def write_provenance(self, dataset, meta, test_run):
        prov = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset.dataset_name,
            "models_processed": meta,
            "test_run": test_run,
            "layer_fractions": LAYER_FRACTIONS
        }
        prefix = OutputConfig.get_path(test_run)
        blob = self.client.bucket(dataset.bucket).blob(f"{prefix}/provenance.json")
        blob.upload_from_string(json.dumps(prov, indent=2))
        print(f"   ✓ Provenance saved")