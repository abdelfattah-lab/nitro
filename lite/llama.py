from rewritten_models import ModelArgs
from pathlib import Path
import openvino as ov
import torch

from typing import Optional

class Llama:
    def __init__(self, model_path: Path | str, device:str):

        # OpenVINO model as the backend
        self.model = ov.compile_model(model_path, device)

        # KV-cache instantiations
        self.k_cache = torch.zeros(eval(self.model.input("cache_k").get_shape().to_string()))
        self.v_cache = torch.zeros(eval(self.model.input("cache_v").get_shape().to_string()))
        self.mask = torch.full(eval(self.model.input("mask").get_shape().to_string()), float("-inf"))
        pass

    @classmethod
    def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        """
        precompute_freqs_cis, outputting as real numbers
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float16)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        freqs_cis = torch.view_as_real(freqs_cis)
        return freqs_cis

    def _prefill() -> None:
        pass

    def generate(self,
                 prompt: str,
                 max_new_tokens: Optional[int]
                 ) -> None:
        
        pass
