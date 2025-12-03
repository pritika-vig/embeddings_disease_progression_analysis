import torch
import timm
import re
from .base import PathologyModelAdapter
from timm.data import resolve_data_config, create_transform

class TimmViTAdapter(PathologyModelAdapter):
    """
    Shared Base Class for standard TIMM models (DINOv2, UNI2, Virchow, GigaPath).
    """
    def __init__(self, device, hf_hub_id, **timm_kwargs):
        super().__init__(device)
        self.hf_hub_id = hf_hub_id
        
        # FORCE unpooled output (no classifier, no pooling)
        timm_kwargs['num_classes'] = 0
        timm_kwargs['global_pool'] = '' 
        self.timm_kwargs = timm_kwargs
        
        self._num_registers = 0
        self._patch_start = 1
        self._embed_dim = None

    def load(self):
        self.model = timm.create_model(self.hf_hub_id, pretrained=True, **self.timm_kwargs)
        self.model.eval().to(self.device)
        self._detect_token_layout()
        
        # Dynamic Dim Check
        if self._embed_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(self.device)
                feat = self.forward_features(dummy) 
                
                # Check for Tuple/Tensor
                if isinstance(feat, tuple): 
                    seq = feat[1]
                else: 
                    seq = feat
                    
                if len(seq.shape) != 3:
                     raise RuntimeError(f"Model {self.hf_hub_id} returned {seq.shape}, expected 3D sequence")
                self._embed_dim = seq.shape[-1]

    def _detect_token_layout(self):
        # 1. Detect Registers via Attributes
        if hasattr(self.model, 'num_register_tokens'):
            self._num_registers = int(self.model.num_register_tokens)
        elif hasattr(self.model, 'register_tokens') and hasattr(self.model.register_tokens, 'shape'):
            self._num_registers = self.model.register_tokens.shape[1]
        elif hasattr(self.model, 'reg_tokens'):
            self._num_registers = int(self.model.reg_tokens)
            
        # 2. Detect Registers via Name Parsing (Regex)
        if self._num_registers == 0:
            names_to_check = [self.hf_hub_id]
            if hasattr(self.model, 'default_cfg'):
                names_to_check.append(self.model.default_cfg.get('architecture', ''))
            
            for name in names_to_check:
                match = re.search(r'reg(\d+)', name)
                if match:
                    self._num_registers = int(match.group(1))
                    break
        
        # 3. Detect Registers via Known Model Overrides
        if self._num_registers == 0:
            name_lower = self.hf_hub_id.lower()
            if 'uni2' in name_lower:
                self._num_registers = 8
            elif 'virchow' in name_lower: # Virchow & Virchow2
                self._num_registers = 4

        # 4. Detect Patch Start
        if hasattr(self.model, 'num_prefix_tokens'):
            self._patch_start = self.model.num_prefix_tokens
        else:
            self._patch_start = 1 + self._num_registers

        # Ensure that (CLS + Registers) matches the actual Prefix count
        expected_patch_start = 1 + self._num_registers
        if self._patch_start != expected_patch_start:
            raise ValueError(
                f"Token Consistency Error for {self.hf_hub_id}: "
                f"Model has {self._patch_start} prefix tokens, "
                f"but we detected {self._num_registers} registers "
                f"(1 CLS + {self._num_registers} != {self._patch_start}). "
                f"Please update explicit detection logic."
            )

    @property
    def num_registers(self) -> int:
        return self._num_registers

    @property
    def patch_start_index(self) -> int:
        return self._patch_start

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None: raise RuntimeError("Model not loaded")
        return self._embed_dim

    def get_transform(self):
        if self.model:
            cfg = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        else:
            model = timm.create_model(self.hf_hub_id, pretrained=False, **self.timm_kwargs)
            cfg = resolve_data_config(model.pretrained_cfg, model=model)
        return create_transform(**cfg)

    def get_blocks(self):
        return self.model.blocks

    def forward_features(self, images):
        output = self.model.forward_features(images)
        
        if len(output.shape) == 2:
             raise RuntimeError(f"Model returned 2D tensor {output.shape}. Check if pooling is disabled.")
             
        # Return (CLS, Sequence)
        return output[:, 0, :], output