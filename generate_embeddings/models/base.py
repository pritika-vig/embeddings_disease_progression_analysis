from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Dict

class PathologyModelAdapter(ABC):
    def __init__(self, device: str):
        self.device = device
        self.model = None

    @abstractmethod
    def load(self): pass

    @abstractmethod
    def get_transform(self): pass

    @abstractmethod
    def get_blocks(self) -> nn.ModuleList: pass

    @property
    @abstractmethod
    def embed_dim(self) -> int: pass
    
    @property
    @abstractmethod
    def default_batch_size(self) -> int: pass

    # --- EXPLICIT TOKEN INDICES ---
    
    @property
    def cls_token_index(self) -> int:
        """Index of the CLS token in the sequence. Default 0."""
        return 0

    @property
    @abstractmethod
    def patch_start_index(self) -> int:
        """Start index of the image patches."""
        pass

    @property
    def num_registers(self) -> int:
        """Number of register tokens. Default 0."""
        return 0
        
    @property
    def reg_start_index(self) -> int:
        """
        Start index of register tokens. 
        Returns -1 if num_registers is 0.
        """
        if self.num_registers == 0: return -1
        # Default logic: Registers come immediately before patches
        return self.patch_start_index - self.num_registers

    @abstractmethod
    def forward_features(self, images) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
        
    def get_token_info(self) -> Dict[str, int]:
        """Helper for validation/logging"""
        return {
            "cls_idx": self.cls_token_index,
            "n_regs": self.num_registers,
            "reg_start": self.reg_start_index,
            "patch_start": self.patch_start_index
        }

    def cleanup(self):
        if self.model:
            del self.model
            torch.cuda.empty_cache()