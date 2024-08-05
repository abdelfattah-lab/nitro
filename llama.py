from base import LLMBase, OVWrapper
from pytorch_model import LlamaModel
from pytorch_model.llama.config import LlamaArgs
from pytorch_model.utils.model_utils import precompute_freqs_cis_rect

from converter import Converter, ConversionConfig
import gc

import openvino as ov
import torch
import os
from pathlib import Path
import time

# These two imports are essential to ensure that the tokenizers can be imported.
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
from modifiers import parse_and_rename_layers, make_stateful, conversion_wrapper, get_shape_dict
import re

class LlamaWrapper(OVWrapper):
    def __init__(self,
                 llm_dir: Path | str,
                 device: str,
                 num_chunks:int,
                 verbose:bool,
                 warm_up:bool = True,
                 compile:bool = True):
        super().__init__(llm_dir, device, num_chunks, verbose, warm_up, compile)

    def setup_transitions(self):
        # Connections
        # TODO: INCOMPLETE
        self._print_if_verbose("Setting up transitions...")
        for i in range(1, len(self.requests)):
            req_1 = self.requests[i-1]
            req_2 = self.requests[i]

            # Cascading [x] connections
            output = req_1.get_output_tensor(0)
            req_2.set_tensor('x', output)
    
    def __call__(self,
                 parallel_inputs:dict[str, torch.Tensor],
                 series_inputs:dict[str, torch.Tensor]
                 ) -> dict[str, torch.Tensor]:
        # TODO: CONVERT FROM SYNCHRONOUS TO ASYNCHRONOUS.
        # Some modifications
        inputs = {}
        inputs.update(parallel_inputs)
        inputs.update(series_inputs)
        for i, model in enumerate(self.models):
            output = model(inputs)
            if "x" in output:
                inputs["x"] = output["x"]
        return output["logit"]
        


class Llama(LLMBase):
    def __init__(self, model_dir: Path | str,
                 args:LlamaArgs,
                 count:int,
                 device:str,
                 compile:bool=True,
                 compress:bool=True,
                 verbose:bool=False
                 ):
        
        super().__init__(model_dir, args, count, device, compile, compress, verbose, LlamaWrapper)
        # Custom inputs
        self._freqs_cis = self._precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta)
        self.freqs_cis = self._freqs_cis[0:1]
        self.mask = torch.full([args.max_batch_size, args.n_heads, 1, args.max_seq_len], float('-inf'))
    
    @classmethod
    def from_pretrained(cls,
                        model_dir: Path | str,
                        max_batch_size: int,
                        max_seq_len: int,
                        chunk_size: int,
                        inference_size: int = 1,
                        export:bool = False,
                        *args, **kwargs
                        ) -> "Llama":
        """
        Generates the Llama model from source code.
        """
        model_args = LlamaArgs()

        model_args.chunk_size = chunk_size
        model_args.max_seq_len = max_seq_len
        model_args.max_batch_size = max_batch_size
        model_args.inference_size = inference_size

        if not os.path.exists(model_dir) or export:
            # TODO: need to obtain meta-llama
            converter = Converter("meta-llama/Meta-Llama-3-8B", model_dir, model_args)
            converter.initialize_model()
            converter.convert_chunks()

            del converter # to save space
        
        count = 0
        llm_dir = Path(model_dir) / "model"
        pattern = re.compile(r'^(\d+)\.xml$')
        for filename in os.listdir(llm_dir):
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                if number > count:
                    count = number
        count += 1
        print(count)

        gc.collect() # the collector object takes up a lot of space; make sure its cleared first.

        print("Creating Llama object")
        return Llama(model_dir, model_args, count, **kwargs)

    def _precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        """
        precompute_freqs_cis, outputting as real numbers
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float16)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        freqs_cis = torch.view_as_real(freqs_cis)
        return freqs_cis

    def _iterate(self, token:int, step:int) -> torch.Tensor:
        """
        Performs one iteration of the LLM: Inputs a token, returns the logits.
        """
        self._update_freqs_cis(step)
        self._update_mask(step)
        
        self.parallel_inputs["freqs_cis"] = self.freqs_cis
        self.parallel_inputs["mask"] = self.mask
        self.series_inputs["x"] = token

        output = self.model(self.parallel_inputs, self.series_inputs)
        return output
    

    def _update_mask(self, step:int) -> None:
        """
        Updates [self.mask] - must be updated with each new token.
        """
        self.mask[:,:,:,-1-step:] = 0.0

    def _update_freqs_cis(self, step:int) -> None:
        self.freqs_cis = self._freqs_cis[step:step+1]

if __name__ == "__main__":

    llama = Llama.from_pretrained(model_dir="npu_model", max_batch_size=1,
                                  max_seq_len=128, chunk_size=16, export=True,
                                  device="NPU",
                                  compile=True,
                                  compress=False,
                                  verbose=True)
    
    output = llama.generate(prompt=["I was wondering why you"],
                            max_new_tokens=12)

    print(output)