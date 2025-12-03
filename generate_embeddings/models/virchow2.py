import torch
from timm.layers import SwiGLUPacked
from .common import TimmViTAdapter

class Virchow2Adapter(TimmViTAdapter):
    def __init__(self, device):
        super().__init__(device, "hf-hub:paige-ai/Virchow2", 
                         mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    @property
    def default_batch_size(self): return 24