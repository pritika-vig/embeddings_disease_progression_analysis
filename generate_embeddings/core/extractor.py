import torch
import numpy as np
from typing import Dict
from models.base import PathologyModelAdapter

class LayerExtractor:
    """
    Extracts embeddings at specific depth fractions.
    Separates output into:
      1. CLS Token
      2. Mean Register Tokens (if present)
      3. Mean Patch Tokens
    """
    def __init__(self, adapter: PathologyModelAdapter, layer_fractions: list):
        self.adapter = adapter
        self.layer_fractions = layer_fractions
        self.hooks = []
        self.activations = {}
        
        self.blocks = self.adapter.get_blocks()
        self.n_blocks = len(self.blocks)
        
        # Token Indices
        self.cls_idx = self.adapter.cls_token_index
        self.patch_start = self.adapter.patch_start_index
        self.reg_start = self.adapter.reg_start_index
        self.n_regs = self.adapter.num_registers
        
        self._setup_hooks()
        
    def _setup_hooks(self):
        # We'll store the mapping to expose it later for provenance
        self.layer_map = {}
        
        for frac in self.layer_fractions[:-1]:
            idx = min(int(np.round(frac * self.n_blocks)) - 1, self.n_blocks - 1)
            idx = max(0, idx)
            self.layer_map[str(frac)] = int(idx)
            self.blocks[idx].register_forward_hook(self._make_hook(f'block_{frac}'))
            
        # Hook final block
        # CHANGE 1: Use '1.0' for the map
        self.layer_map['1.0'] = self.n_blocks - 1
        self.blocks[-1].register_forward_hook(self._make_hook('final_block'))

    def get_provenance_meta(self) -> dict:
        return {
            "cls_token_index": self.cls_idx,
            "num_registers": self.n_regs,
            "register_start_index": self.reg_start,
            "patch_start_index": self.patch_start,
            "total_blocks": self.n_blocks,
            "layer_depth_mapping": self.layer_map
        }

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple): output = output[0]
            self.activations[name] = output.detach()
        return hook

    def _process_sequence(self, sequence: torch.Tensor, depth_key: str, results: dict):
        # 1. CLS Token
        results[f'cls_{depth_key}'] = sequence[:, self.cls_idx, :].cpu().numpy()
        
        # 2. Register Tokens
        if self.n_regs > 0:
            regs = sequence[:, self.reg_start : self.patch_start, :]
            results[f'register_mean_{depth_key}'] = regs.mean(dim=1).cpu().numpy()
        
        # 3. Patch Tokens
        if sequence.size(1) > self.patch_start:
            patches = sequence[:, self.patch_start:, :]
            results[f'patch_mean_{depth_key}'] = patches.mean(dim=1).cpu().numpy()
        else:
            results[f'patch_mean_{depth_key}'] = sequence[:, self.cls_idx, :].cpu().numpy()

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> Dict[str, np.ndarray]:
        self.activations = {}
        
        # 1. Run Forward Pass (Get Projected/Official Embedding)
        final_projected = self.adapter.forward_features(images)
        if isinstance(final_projected, tuple):
             final_projected = final_projected[0]

        embeddings = {}
        
        # 2. Store Official Embedding
        embeddings['final_embedding'] = final_projected.cpu().numpy()
        
        # 3. Process Intermediate Layers
        for frac in self.layer_fractions[:-1]:
            key = f'block_{frac}'
            if key in self.activations:
                self._process_sequence(self.activations[key], str(frac), embeddings)

        # 4. Process Final Layer
        if 'final_block' in self.activations:
            self._process_sequence(self.activations['final_block'], '1.0', embeddings)

        # Normalize Everything
        for k, v in embeddings.items():
            embeddings[k] = np.array([self._l2_normalize(x) for x in v])
            
        return embeddings

    def _l2_normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}