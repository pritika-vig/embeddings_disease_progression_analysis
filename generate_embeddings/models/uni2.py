import torch
from timm.layers import SwiGLUPacked
from .common import TimmViTAdapter

# https://huggingface.co/MahmoodLab/UNI2-h

class Uni2Adapter(TimmViTAdapter):
    def __init__(self, device):
        kwargs = {
            'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
            'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 5.33334,
            'num_classes': 0, 'no_embed_class': True, 'mlp_layer': SwiGLUPacked,
            'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True
        }
        super().__init__(device, "hf-hub:MahmoodLab/UNI2-h", **kwargs)

    @property
    def default_batch_size(self): return 24