import torch
from .common import TimmViTAdapter

class DinoV2Adapter(TimmViTAdapter):
    def __init__(self, device):
        super().__init__(
                    device, 
                    'vit_large_patch14_reg4_dinov2.lvd142m', 
                    img_size=224, 
                    dynamic_img_size=True
                )

    @property
    def default_batch_size(self): return 32