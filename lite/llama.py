from config import ModelArgs
from pathlib import Path
import openvino as ov
import torch
import torch.nn as nn
from typing import Optional
import numpy as np

# These two imports are essential to ensure that the tokenizers can be imported.
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer

class Llama:
    def __init__(self,
                 model_path: Path | str,
                 device:str,
                 args: ModelArgs,
                 tokenizer:str="openvino_tokenizer.xml",
                 detokenizer:str="openvino_detokenizer.xml"
                 ):

        # OpenVINO models as the backend
        core = ov.Core()
        self.model = ov.compile_model(model_path, device_name=device)
        self.tokenizer = core.compile_model(tokenizer, "CPU")
        self.detokenizer = core.compile_model(detokenizer, "CPU")

        # KV-cache, mask instantiations - these are inputs
        self.k_cache = torch.zeros(eval(self.model.input("cache_k").get_shape().to_string()))
        self.v_cache = torch.zeros(eval(self.model.input("cache_v").get_shape().to_string()))
        self.mask = torch.full(eval(self.model.input("mask").get_shape().to_string()), float("-inf"))

        # Token embeddings, if done outside the model
        # self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.freqs_cis = self._precompute_freqs_cis(args.dim // args.n_heads,
                                                    args.max_seq_len * 2,
                                                    args.rope_theta)

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
        for i, token in enumerate(tokens):
            self._update_mask(i)
            fs = self.freqs_cis[i:i+1]
            # x = embedding(tok_embeddings, torch.tensor(token).view(1, -1))
            output = self.model({
                "cache_v": self.v_cache,
                "cache_k": self.k_cache,
                "mask": self.mask,
                "freqs_cis": fs,
                "x": torch.tensor([[token]])
            })
            self.k_cache = output[1]
            self.v_cache = output[2]
        

    def _decode(self, tokens:list[int], first_token:torch.Tensor, max_new_tokens:int) -> list[int]:
        """
        Decode stage of text generation. Modifies [tokens] in place and returns
        it.
        
        Parameters:
        tokens (list[int]) : tokens from the prompt.
        first_token (int) : first token in the decode stage, which is the last token in the input sequence.
        max_new_tokens (int) : number of new tokens to generate.
        
        Returns:
        tokens: list[int] : complete token sequence.
        """
        next_token = first_token
        start_idx = len(tokens)
        for i in range(start_idx, start_idx+max_new_tokens):
            tokens.append(next_token)

            self._update_mask(i)
            fs = self.freqs_cis[i:i+1]
            # x = embedding(tok_embeddings, torch.tensor(t).view(1, -1)) # HARD-CODED TO BE B = L = 1: [B,L,DIM]
            output = self.model({
                "cache_v": self.v_cache,
                "cache_k": self.k_cache,
                "mask": self.mask,
                "freqs_cis": fs,
                "x": torch.tensor([[next_token]])
            })
            logits = output[0]
            self.k_cache = output[1]
            self.v_cache = output[2]
            next_token = np.argmax(logits.squeeze()) # TODO: ADD OPTION FOR VARIATION
    
        return tokens
    
    def _update_mask(self, step:int) -> None:
        """
        Updates [self.mask] - must be updated with each new token.
        """
        self.mask[:,:,:,-1-step:] = 0

    def _generate_tokens(self, prompt) -> str:
        """
        Converts the list of prompts into a Python list of tokens.
        """
        return list(self.tokenizer(prompt)["input_ids"].squeeze())

    def generate(self,
                 prompt: list[str],
                 max_new_tokens: Optional[int]
                 ) -> None:
        tokens = self._generate_tokens(prompt)
        next_token = tokens.pop()

        # Prefill
        self._prefill(tokens)

        # Generate
        tokens = self._decode(tokens, next_token, max_new_tokens)

        # Detokenizer
        outputs = self.detokenizer(np.array(tokens).reshape(1, -1))
        return outputs["string_output"]

llama = Llama("llama.xml", "CPU", ModelArgs())
print(llama.generate(["Hello, how are you?"], 20))