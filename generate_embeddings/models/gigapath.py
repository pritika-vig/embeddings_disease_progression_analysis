from .common import TimmViTAdapter

# https://huggingface.co/prov-gigapath/prov-gigapath 

class GigaPathAdapter(TimmViTAdapter):
    def __init__(self, device):
        super().__init__(device, "hf-hub:prov-gigapath/prov-gigapath")

    @property
    def default_batch_size(self): return 24