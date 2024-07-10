from model.config import ModelArgs
from pathlib import Path
import openvino as ov
import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import nncf
import time

# These two imports are essential to ensure that the tokenizers can be imported.
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer

class Llama:
    def __init__(self,
                 model_path: Path | str,
                 device:str,
                 args: ModelArgs,
                 tokenizer:str="openvino_model/openvino_tokenizer.xml",
                 detokenizer:str="openvino_model/openvino_detokenizer.xml",
                 compile:bool=True,
                 compress:bool=True,
                 embedding:bool=True,
                 verbose:bool=False
                 ):
        self.verbose = verbose
        self.device = device
        self.n_layers = args.n_layers

        start = time.time()
        # OpenVINO models as the backend
        self._print_if_verbose("Importing model...")
        core = ov.Core()
        self.model = core.read_model(model_path)
        if compress:
            self.model = nncf.compress_weights(self.model, mode=nncf.CompressWeightsMode.INT8_ASYM)
        if compile:
            self._print_if_verbose(f"Compiling model to {self.device}...")
            self.model = core.compile_model(self.model, self.device)
        self._print_if_verbose("Compiling tokenizers...")
        self.tokenizer = core.compile_model(tokenizer, "CPU")
        self.detokenizer = core.compile_model(detokenizer, "CPU")

        # KV-cache, mask instantiations - these are inputs
        self._print_if_verbose("Instantiating Parameters...")
        print(self.model)
        self.k_cache = {}
        self.v_cache = {}
        for i in range(args.n_layers):
            self.k_cache[f"cache_k_{i}"] = torch.zeros(eval(self.model.input("cache_k_1").get_shape().to_string()))
            self.v_cache[f"cache_v_{i}"] = torch.zeros(eval(self.model.input("cache_k_1").get_shape().to_string()))
        self.mask = torch.full(eval(self.model.input("mask").get_shape().to_string()), float("-inf"))

        # Token embeddings, if done outside the model.
        if embedding:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim).eval() # THIS THING NEEDS TO BE LOADED IN AS WELL.
        else:
            self.tok_embeddings = lambda x : x

        self.freqs_cis = self._precompute_freqs_cis(args.dim // args.n_heads,
                                                    args.max_seq_len * 2,
                                                    args.rope_theta)
    
        elapsed = time.time() - start
        self._print_if_verbose(f"Finished pre-processing. Elapsed time: {elapsed:.4f}")

    def compile(self, device:str=None):
        if device is not None:
            self.device = device
        start = time.time()
        self._print_if_verbose(f"Compiling to {self.device}...")

        # Experimental properties
        core = ov.Core()

        self.model = core.compile_model(self.model, self.device)

        self._print_if_verbose(f"Compiled in {(time.time() - start):.4f} seconds")
        pass

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

    def _prefill(self, tokens:list[int]) -> None:
        """
        Pre-fill stage of text generation. Current implementation is to feed each
        token one-by-one, like the decode stage, while ignoring the output token.
        
        k_cache, v_cache, mask are all updated.
        """
        average_time = 0
        
        for i, token in enumerate(tokens):
            start = time.time()
            self._update_mask(i)
            fs = self.freqs_cis[i:i+1]

            # Preparing the inputs
            inputs = {
                "mask": self.mask,
                "freqs_cis": fs,
                "x": torch.tensor([[token]])
            }
            inputs.update(self.k_cache)
            inputs.update(self.v_cache)

            # Running inference
            output = self.model(inputs)
            
            # Updating the KV-caches: 1-32 are the k-caches, 33-64 are the v-caches
            for j in range(0, self.n_layers):
                self.k_cache[f"cache_k_{j}"] = output[f"cache_k_{j}_out"]
                self.v_cache[f"cache_v_{j}"] = output[f"cache_v_{j}_out"]

            elapsed = time.time() - start
            average_time = (average_time * i + elapsed) / (i + 1)
            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")

    def _decode(self, tokens:list[int], first_token:torch.Tensor, max_new_tokens:int) -> list[int]:
        """
        Decode stage of text generation. Modifies [tokens] in place and returns
        it.
        
        Parameters:
        tokens (list[int])      : tokens from the prompt.
        first_token (int)       : first token in the decode stage, which is the last token in the input sequence.
        max_new_tokens (int)    : number of new tokens to generate.
        
        Returns:
        tokens: list[int]       : complete token sequence.
        """
        next_token = first_token
        start_idx = len(tokens)

        average_time = 0
        for i in range(start_idx, start_idx+max_new_tokens):
            start = time.time()

            tokens.append(next_token)
            self._update_mask(i)
            fs = self.freqs_cis[i:i+1]

            # Preparing inputs
            inputs = {
                "mask": self.mask,
                "freqs_cis": fs,
                "x": torch.tensor([[next_token]])
            }
            inputs.update(self.k_cache)
            inputs.update(self.v_cache)

            # Running inference
            output = self.model(inputs)

            # Logits
            logits = output["logits"]
            
            # Updating the KV-caches: 1-32 are the k-caches, 33-64 are the v-caches
            for j in range(0, self.n_layers):
                self.k_cache[f"cache_k_{j}"] = output[f"cache_k_{j}_out"]
                self.v_cache[f"cache_v_{j}"] = output[f"cache_v_{j}_out"]
            next_token = np.argmax(logits.squeeze()) # TODO: ADD OPTION FOR VARIATION
            # print(logits.shape)

            elapsed = time.time() - start
            average_time = (average_time * (i - start_idx) + elapsed) / (i - start_idx + 1)
            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")
    
        return tokens

    def _print_if_verbose(self, *text) -> None:
        if self.verbose:
            print(*text)
    
    def _update_mask(self, step:int) -> None:
        """
        Updates [self.mask] - must be updated with each new token.
        """
        self.mask[:,:,:,-1-step] = 0

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


    llama = Llama(model_path="openvino_model/llama.xml",
                  device="GPU",
                  args=ModelArgs(n_layers=32, max_batch_size=1, max_seq_len=128),
                  compile=False,
                  embedding=False,
                  verbose=True)
    
    llama_backend = llama.model

    # raise Exception
    llama.compile()

    print(llama.generate(["What is going on"], max_new_tokens=30))