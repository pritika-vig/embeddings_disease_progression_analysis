import torch
import gc
from google.cloud import storage
from datetime import datetime
from config import DATA_ROOT
from models import get_model_adapter

class DryRunValidator:
    def __init__(self, device='cuda'):
        self.device = device

    def validate_dataset(self, dataset_config):
        print(f"📁 Checking dataset: {dataset_config.dataset_name}")
        path = DATA_ROOT / dataset_config.dataset_name / "train"
        if not path.exists():
            print(f"   ❌ Path not found: {path}")
            return False
        
        found = 0
        for cls in dataset_config.classes:
            if (path / cls).exists(): found += 1
            else: print(f"   ❌ Missing class dir: {cls}")
        
        print(f"   ✓ Found {found}/{len(dataset_config.classes)} classes")
        return found > 0

    def validate_gcs(self, bucket_name):
        print(f"☁️  Checking GCS: gs://{bucket_name}")
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"test_write_{datetime.now().timestamp()}.txt")
            blob.upload_from_string("test")
            blob.delete()
            print("   ✓ Read/Write access confirmed")
            return True
        except Exception as e:
            print(f"   ❌ GCS Error: {e}")
            return False

    def validate_model(self, model_name):
        print(f"🤖 Checking Model: {model_name}")
        try:
            adapter = get_model_adapter(model_name, self.device)
            adapter.load()
            
            # INSPECT INDICES
            info = adapter.get_token_info()
            print(f"   Token Layout:")
            print(f"     • CLS Index: {info['cls_idx']}")
            print(f"     • Registers: {info['n_regs']} (Start: {info['reg_start']})")
            print(f"     • Patches Start: {info['patch_start']}")
            print(f"     • Embed Dim: {adapter.embed_dim}")
            print(f"     • Blocks: {len(adapter.get_blocks())}")

            # DUMMY INFERENCE
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            if model_name == 'musk': 
                dummy = torch.randn(1, 3, 384, 384).to(self.device)
            
            cls_out, seq_out = adapter.forward_features(dummy)
            
            # VERIFY DIMENSIONS
            if cls_out.shape[-1] != adapter.embed_dim:
                print(f"   ❌ Dimension Mismatch! Expected {adapter.embed_dim}, got {cls_out.shape[-1]}")
                return False
                
            # VERIFY SLICING
            total_tokens = seq_out.shape[1]
            if total_tokens <= info['patch_start']:
                print(f"   ❌ Sequence too short ({total_tokens}) for patch start ({info['patch_start']})")
                return False
            else:
                patch_count = total_tokens - info['patch_start']
                print(f"   ✓ Layout Valid. Found {patch_count} patches.")
                
            adapter.cleanup()
            return True
            
        except Exception as e:
            print(f"   ❌ Model failed: {e}")
            import traceback
            traceback.print_exc()
            return False