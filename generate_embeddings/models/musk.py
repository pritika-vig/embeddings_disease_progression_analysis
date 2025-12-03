import torch
import torchvision.transforms as T
from .base import PathologyModelAdapter
import musk
import musk.modeling 
from musk import utils
from timm.models import create_model

# https://huggingface.co/xiangjx/musk 

class MuskAdapter(PathologyModelAdapter):
    def __init__(self, device):
        super().__init__(device)
        self._embed_dim = None

    def get_transform(self):
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        return T.Compose([
            T.Resize(384, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(384),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    def load(self):
        self.model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", self.model, 'model|module', '')
        self.model.eval().to(self.device)
        
        if self._embed_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 384, 384).to(self.device)
                val, _ = self.forward_features(dummy)
                self._embed_dim = val.shape[-1]

    @property
    def patch_start_index(self) -> int:
        return 1

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def default_batch_size(self) -> int:
        return 8

    def get_blocks(self):
        return self.model.beit3.encoder.layers

    def forward_features(self, images):
        output = self.model(image=images, with_head=False, out_norm=True, return_global=True)
        if isinstance(output, tuple): final_cls = output[0]
        else: final_cls = output
            
        sequence = self.model(image=images, with_head=False, out_norm=False, return_global=False)
        
        # --- FIX: Unpack tuple if needed ---
        if isinstance(sequence, (tuple, list)):
            sequence = sequence[0]
            
        return final_cls, sequence