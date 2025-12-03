from .dinov2 import DinoV2Adapter
from .uni2 import Uni2Adapter
from .gigapath import GigaPathAdapter
from .virchow2 import Virchow2Adapter
from .conch import ConchAdapter
from .musk import MuskAdapter

REGISTRY = {
    "dinov2": DinoV2Adapter,
    "uni2": Uni2Adapter,
    "gigapath": GigaPathAdapter,
    "virchow2": Virchow2Adapter,
    "conch": ConchAdapter,
    "musk": MuskAdapter
}

def get_model_adapter(name: str, device: str):
    if name not in REGISTRY:
        raise ValueError(f"Model {name} not found. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](device)