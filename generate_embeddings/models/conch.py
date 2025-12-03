import torch
import torchvision.transforms as T
from .base import PathologyModelAdapter

# https://huggingface.co/MahmoodLab/CONCH 

class ConchAdapter(PathologyModelAdapter):
    def __init__(self, device):
        super().__init__(device)
        self._embed_dim = None

    def get_transform(self):
        return T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def load(self):
        from conch.open_clip_custom import create_model_from_pretrained
        self.model, _ = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/CONCH')
        self.model.eval().to(self.device)
        
        # Dynamic Dim Check
        if self._embed_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(self.device)
                val = self.forward_features(dummy)
                self._embed_dim = val.shape[-1]

    @property
    def patch_start_index(self) -> int:
        return 1

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def default_batch_size(self) -> int:
        return 48

    def get_blocks(self):
        return self.model.visual.trunk.blocks

    def forward_features(self, images):
        # 1. Run the full model once
        # proj_contrast=False: Return [B, 512] visual embedding (not projected to text space)
        # normalize=False: We let LayerExtractor handle the final L2 normalization
        final_embedding = self.model.encode_image(
            images, 
            proj_contrast=False, 
            normalize=True
        )
        
        return final_embedding