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
        self.cls_idx = self.adapter.cls_token_index
        
        # Token slicing logic
        self.n_registers = self.adapter.num_registers
        # Sequence layout: [CLS, Reg_1...Reg_N, Patch_1...Patch_M]
        self.patch_start_idx = 1 + self.n_registers
        
        self._setup_hooks()
        
    def _setup_hooks(self):
            # We'll store the mapping to expose it later for provenance
            self.layer_map = {}
            
            for frac in self.layer_fractions[:-1]:
                # Calculate the exact block index
                idx = min(int(np.round(frac * self.n_blocks)) - 1, self.n_blocks - 1)
                idx = max(0, idx)
                
                # Save mapping: fraction -> actual block index (0-based)
                self.layer_map[str(frac)] = int(idx)
                
                self.blocks[idx].register_forward_hook(self._make_hook(f'block_{frac}'))
                
            # Add final block to map
            self.layer_map['final'] = self.n_blocks - 1
            self.blocks[-1].register_forward_hook(self._make_hook('final_block'))

    def _make_hook(self, name):
        def hook(module, input, output):
            # Unwrap tuple outputs if necessary
            if isinstance(output, tuple): output = output[0]
            self.activations[name] = output.detach()
        return hook

    def _process_sequence(self, sequence: torch.Tensor, depth_key: str, results_dict: dict):
        """Helper to slice the sequence [B, N, D] into component parts"""
        # 1. CLS Token
        results_dict[f'cls_{depth_key}'] = sequence[:, 0, :].cpu().numpy()
        
        # 2. Register Tokens (Mean Pool)
        if self.n_registers > 0:
            reg_tokens = sequence[:, 1 : self.patch_start_idx, :]
            results_dict[f'register_mean_{depth_key}'] = reg_tokens.mean(dim=1).cpu().numpy()
        
        # 3. Patch Tokens (Mean Pool)
        if sequence.size(1) > self.patch_start_idx:
            patch_tokens = sequence[:, self.patch_start_idx:, :]
            results_dict[f'patch_mean_{depth_key}'] = patch_tokens.mean(dim=1).cpu().numpy()
        else:
            # Fallback if no patches found (should not happen in valid data)
            results_dict[f'patch_mean_{depth_key}'] = sequence[:, 0, :].cpu().numpy()

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> Dict[str, np.ndarray]:
        self.activations = {}
        
        # Run Forward Pass
        # We ignore return values here because we rely on the 'final_block' hook
        # to capture the exact state of the final layer.
        self.adapter.forward_features(images)
        
        embeddings = {}
        
        # Process Intermediate Layers
        for frac in self.layer_fractions[:-1]:
            key = f'block_{frac}'
            if key in self.activations:
                self._process_sequence(self.activations[key], str(frac), embeddings)

        # Process Final Layer
        if 'final_block' in self.activations:
            self._process_sequence(self.activations['final_block'], 'final', embeddings)

        # L2 Normalize all embeddings
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

    def get_provenance_meta(self) -> dict:
            """Return exact runtime configuration for provenance logging"""
            return {
                "cls_token_index": self.cls_idx,
                "num_registers": self.n_registers,
                "patch_start_index": self.patch_start_idx,
                "total_blocks": self.n_blocks,
                "layer_depth_mapping": self.layer_map
            }