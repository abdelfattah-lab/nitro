from model.llama.rewritten_models import RMSNorm
from base import Args, LLMBase, OVWrapper

import openvino as ov
import torch
import torch.nn as nn
import numpy as np
import nncf 
import os

from typing import Optional
from pathlib import Path
import time
import json

# These two imports are essential to ensure that the tokenizers can be imported.
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer

from model.llama.rewritten_models import Transformer
from model.llama.helpers import precompute_freqs_cis_rect
from modifiers import parse_and_rename_layers, make_stateful, conversion_wrapper, get_shape_dict
import re
                
class LlamaConfig:
    def __init__(self):

        self.model = "meta-llama/Meta-Llama-3-8B"

        self.dim = 4096
        self.n_layers = 32
        self.n_heads = 32
        self.n_kv_heads = 8
        self.vocab_size = 128256
        self.multiple_of = 1024
        self.ffn_dim_multiplier = 1.3
        self.norm_eps = 5e-5
        self.rope_theta = 500000
        self.max_batch_size = 1
        self.max_seq_len = 128

        self.chunk_size = 16

class Llama(LLMBase):
    def __init__(self, model_dir: Path | str, args:LlamaConfig, count:int, device:str, compile:bool=True, compress:bool=True, verbose:bool=False):
        # TODO
        self.verbose = verbose
        self.device = device

        model_dir = Path(model_dir)
        llm_dir = model_dir / "model"
        token_dir = model_dir / "tokenizer"

        self.args = args
        self.model = OVWrapper(llm_dir, device,
                               num_chunks=count, verbose=verbose, warm_up=False, compile=compile)
    
        # Custom inputs
        self.freqs_cis = self._precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta)
        self.mask = torch.full([args.max_batch_size, args.n_heads, 1, args.max_seq_len], float('-inf'))

        core = ov.Core()
        self.tokenizer = core.compile_model(token_dir / "openvino_tokenizer.xml", "CPU")
        self.detokenizer = core.compile_model(token_dir / "openvino_detokenizer.xml", "CPU")
    
    @classmethod
    def from_pretrained(cls,
                        model_dir: Path | str,
                        max_batch_size: int,
                        max_seq_len: int,
                        chunk_size: int,
                        export:bool = False,
                        *args, **kwargs
                        ) -> "Llama":
        """
        Generates the Llama model from source code.
        """

        args = LlamaConfig()
        args.chunk_size = chunk_size
        args.max_seq_len = max_seq_len
        args.max_batch_size = max_batch_size

        model_dir = Path(model_dir)
        llm_dir = model_dir / "model"

        if export:
            ############################################
            ###     GENERATION OF THE MAIN MODEL     ###
            ############################################

            checkpoint = torch.load(model_dir / "consolidated.00.pth")
            print("Creating PyTorch model...")
            model = Transformer(args, chunk_size=chunk_size)
            print("Loading checkpoint...")
            model.load_state_dict(checkpoint, strict=True)

            count = 0 # counting the number of chunks.

            freqs_cis = precompute_freqs_cis_rect(
                args.dim // args.n_heads,
                args.max_seq_len * 2,
                args.rope_theta
            )

            # Preparing inputs
            L = 1
            example_input = {
                "x"         : torch.randint(0, args.vocab_size, [args.max_batch_size, L]),
                "mask"      : torch.full([args.max_batch_size, args.n_heads, L, args.max_seq_len], float('-inf')),
                "freqs_cis" : freqs_cis[0:L],
                "params"    : {}
            }

            for i in range(args.n_layers):
                example_input["params"][f"cache_k_{i}"] = torch.zeros([args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.dim // args.n_heads])
                example_input["params"][f"cache_v_{i}"] = torch.zeros([args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.dim // args.n_heads])
                
            input_shapes = get_shape_dict(example_input)
            for key in input_shapes["params"]:
                input_shapes[key] = input_shapes["params"][key]
            input_shapes.pop("params")

            #### Chunking embedding layer ####
            print("Converting embedding layer...")
            model.include_embedding, model.include_transformer, model.include_output = True, False, False
            conversion_wrapper(model, count, llm_dir, example_input, input_shapes)
            count += 1

            #### Chunking transformer layers ####
            print("Converting transformer layers...")
            model.include_embedding, model.include_transformer, model.include_output = False, True, False

            for offset in range(0, args.n_layers, args.chunk_size):
                print(f" > Block: {offset}-{offset + args.chunk_size-1}")
                model.offset = offset

                conversion_wrapper(model, count, llm_dir, example_input, input_shapes)
                count += 1

            #### Chunking output layer ####
            model.include_embedding, model.include_transformer, model.include_output = False, False, True
            print("Converting output layer...")

            conversion_wrapper(model, count, llm_dir, example_input, input_shapes)
            count += 1


            #######################################
            ###            TOKENIZER            ###
            #######################################

            import openvino_tokenizers as ot

            print("Generating tokenizers...")
            hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
            ov_tokenizer, ov_detokenizer = ot.convert_tokenizer(hf_tokenizer, with_detokenizer=True, skip_special_tokens=True)
            ov.save_model(ov_tokenizer, model_dir / "tokenizer" / "openvino_tokenizer.xml")
            ov.save_model(ov_detokenizer, model_dir / "tokenizer" / "openvino_detokenizer.xml")
        
        else:
            count = 0
            pattern = re.compile(r'^(\d+)\.xml$')
            for filename in os.listdir(llm_dir):
                match = pattern.match(filename)
                if match:
                    number = int(match.group(1))
                    if number > count:
                        count = number
            count += 1
            print(count)
        return Llama(model_dir, args, count, **kwargs)

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

    def _iterate(self, parallel_inputs, series_inputs) -> torch.Tensor:
        """
        Performs one iteration of the LLM: Inputs a token, returns the logits.
        """
        
        
        output = self.model(parallel_inputs, series_inputs)
        return output

    def _prefill(self, tokens:list[int]) -> None:
        """
        Prefill stage.
        """

        series_inputs = {}
        parallel_inputs = {"freqs_cis" : self.freqs_cis,
                           "mask" : self.mask}
        
        average_time = 0
        for i, token in enumerate(tokens):
            start = time.time()

            # Updating inputs
            series_inputs["x"] = token
            self._update_mask(i)
            parallel_inputs["freqs_cis"] = self.freqs_cis[i:i+1]

            _ = self._iterate(series_inputs, parallel_inputs)
            
            elapsed = time.time() - start
            average_time = (average_time * (i) + elapsed) / (i + 1)

            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")
            
    def _decode(self, tokens:list[int], first_token:torch.Tensor, max_new_tokens:int) -> list[int]:
        """
        Prefill stage.
        """
        token = first_token
        start_idx = len(tokens)

        series_inputs = {}
        parallel_inputs = {"freqs_cis" : self.freqs_cis,
                           "mask" : self.mask}
        
        average_time = 0
        for i in range(start_idx, start_idx+max_new_tokens):
            start = time.time()
            # Updating parallel inputs
            tokens.append(token)
            series_inputs["x"] = token
            self._update_mask(i)
            parallel_inputs["freqs_cis"] = self.freqs_cis[i:i+1]

            output = self._iterate(series_inputs, parallel_inputs)

            token = np.argmax(output.squeeze()) # TODO: ADD OPTION FOR VARIATION

            elapsed = time.time() - start
            average_time = (average_time * (i - start_idx) + elapsed) / (i - start_idx + 1)
            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")

        tokens.append(token)
        return tokens
    

    def _update_mask(self, step:int) -> None:
        """
        Updates [self.mask] - must be updated with each new token.
        """
        self.mask[:,:,:,-1-step:] = 0.0

    def _generate_tokens(self, prompt) -> str:
        """
        Converts the list of prompts into a Python list of tokens.
        """
        return list(self.tokenizer(prompt)["input_ids"].squeeze())

    def generate(self,
                 prompt: list[str],
                 max_new_tokens: Optional[int]
                 ) -> list[str]:
        """
        Runs inference for text generation.

        Parameters:
        prompt (list[str])      : input string prompt.
        max_new_tokens (int)    : number of new tokens to generate.

        Returns:
        tokens (list[str])      : completed text, new tokens with original prompt.
        """
        tokens = self._generate_tokens(prompt)
        next_token = tokens.pop()

        # Prefill
        self._prefill(tokens)

        # Generate
        tokens = self._decode(tokens, next_token, max_new_tokens)

        # Detokenizer
        outputs = self.detokenizer(np.array(tokens).reshape(1, -1))
        return outputs["string_output"]



if __name__ == "__main__":

    llama = Llama.from_pretrained(model_dir="npu_model", max_batch_size=1,
                                  max_seq_len=128, chunk_size=16, export=False,
                                  device="NPU",
                                  compile=True,
                                  compress=False,
                                  verbose=True)

    output = llama.generate(prompt=["I was wondering why you"],
                            max_new_tokens=30)

    print(output)