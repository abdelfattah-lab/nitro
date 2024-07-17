from model.llama.rewritten_models import RMSNorm
from base import Args, LLMBase

import openvino as ov
import torch
import torch.nn as nn
import numpy as np
import nncf 

from typing import Optional
from pathlib import Path
import time
import json

# These two imports are essential to ensure that the tokenizers can be imported.
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer

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

        self.outside_embedding = True
        self.outside_end_layer = True
        self.n_sub_layers = 16

class Llama(LLMBase):
    def __init__(self,
                 model_dir: Path | str,
                 device:str,
                 compile:bool=True,
                 compress:bool=True,
                 verbose:bool=False
                 ):
        
        self.verbose = verbose
        self.device = device

        model_dir = Path(model_dir)
        model_path = model_dir / "model" / "1.xml"
        model_path_2 = model_dir / "model" / "2.xml"
        tokenizer_path = model_dir / "tokenizer" / "openvino_tokenizer.xml"
        detokenizer_path = model_dir / "tokenizer" / "openvino_detokenizer.xml"
        cache_dir = model_dir / "model" / "cache"
        config_dir = model_dir / "config.json"

        # Configuration loading
        with open(config_dir, 'r') as file:
            json_data = file.read()
        args = json.loads(json_data)
        self.n_layers = args["n_layers"]
        self.n_sub_layers = args["n_sub_layers"]
        embedding = args["outside_embedding"]
        end_layer = args["outside_end_layer"]

        start = time.time()
        # OpenVINO models as the backend
        self._print_if_verbose("Importing model...")
        self.core = ov.Core()
        self.core2 = ov.Core()
        self.core3 = ov.Core()
        self.core.set_property({"CACHE_DIR" : cache_dir / "1"})
        self.core2.set_property({"CACHE_DIR" : cache_dir / "2"})
        self.model = self.core.read_model(model_path)
        self.model_2 = self.core2.read_model(model_path_2)
        if compress:
            self.model = nncf.compress_weights(self.model, mode=nncf.CompressWeightsMode.INT8_ASYM)
            self.model_2 = nncf.compress_weights(self.model_2, mode=nncf.CompressWeightsMode.INT8_ASYM)
        if compile:
            self.compile(self.device)
        self._print_if_verbose("Compiling tokenizers...")
        self.tokenizer = self.core3.compile_model(tokenizer_path, "CPU")
        self.detokenizer = self.core3.compile_model(detokenizer_path, "CPU")

        # KV-cache, mask instantiations - these are inputs
        self._print_if_verbose("Instantiating Parameters...")
        self.mask = torch.full(eval(self.model.input("mask").get_shape().to_string()), -float("inf"))
        self.inputs_1 = {}
        self.inputs_2 = {}
        for i in range(self.n_sub_layers):
            self.inputs_1["mask"] = self.mask
            self.inputs_2["mask"] = self.mask
            self.inputs_1[f"cache_k_{i}"] = np.zeros(eval(self.model.input("cache_k_1").get_shape().to_string()))
            self.inputs_1[f"cache_v_{i}"] = np.zeros(eval(self.model.input("cache_k_1").get_shape().to_string()))
            self.inputs_2[f"cache_k_{i}"] = np.zeros(eval(self.model.input("cache_k_1").get_shape().to_string()))
            self.inputs_2[f"cache_v_{i}"] = np.zeros(eval(self.model.input("cache_k_1").get_shape().to_string()))

        # Token embeddings, if done outside the model.
        if embedding or end_layer:
            checkpoint = torch.load(model_dir / "consolidated.00.pth")

            if embedding:
                print("Creating and loading embedding")
                self.tok_embeddings = nn.Embedding(args["vocab_size"], args["dim"])
                self.tok_embeddings.load_state_dict({'weight': checkpoint['tok_embeddings.weight']})
                self.tok_embeddings = torch.compile(self.tok_embeddings, backend="openvino", options={"device" : "NPU"})
            
            if end_layer:
                print("Creating and loading norm and end layer")
                
                self.norm = RMSNorm(args["dim"], eps=args["norm_eps"])
                self.norm.load_state_dict({'weight': checkpoint['norm.weight']})
                self.norm = torch.compile(self.norm, backend="openvino", options={"device" : "CPU"})
                _ = self.norm(torch.randn([1, 1, args["dim"]])) # Warming up the model
                
                self.linear = nn.Linear(args["dim"], args["vocab_size"], bias=False)
                self.linear.load_state_dict({'weight': checkpoint['output.weight']})
                self.linear = torch.compile(self.linear, backend="openvino", options={"device" : "CPU"})
                _ = self.linear(torch.randn([1, 1, args["dim"]])) # Warming up the model

        self.freqs_cis = self._precompute_freqs_cis(args["dim"] // args["n_heads"],
                                                    args["max_seq_len"] * 2,
                                                    args["rope_theta"])
    
        elapsed = time.time() - start
        self._print_if_verbose(f"Finished pre-processing. Elapsed time: {elapsed:.4f}")
    
    @classmethod
    def from_pretrained(cls,
                        model_dir: Path | str,
                        max_batch_size: int,
                        max_seq_len: int,
                        n_sub_layers: int,
                        export:bool = False,
                        *args, **kwargs
                        ) -> "Llama":
        """
        Generates the Llama model from source code.
        """
        if export:
            from model.llama.rewritten_models import Transformer
            from model.llama.helpers import precompute_freqs_cis_rect
            from rename import parse_and_rename_layers
            import re

            ############################################
            ###     GENERATION OF THE MAIN MODEL     ###
            ############################################
            def get_shape_dict(d):
                if isinstance(d, dict):
                    shape_dict = {}
                    for key, value in d.items():
                        shape_dict[key] = get_shape_dict(value)
                    return shape_dict
                elif isinstance(d, torch.Tensor):
                    return d.shape
                else:
                    raise TypeError("Unsupported type: {}".format(type(d)))
                    
            model_dir = Path(model_dir)
            config_dir = model_dir / "config.json"

            with open(config_dir, 'r') as file:
                json_data = file.read()
            args = json.loads(json_data)

            print("Model parameters:")
            print(args)

            DIM = args["dim"]
            B = max_batch_size
            KVH = args["n_kv_heads"]
            NH = args["n_heads"]      
            ML = max_seq_len
            HD = DIM // NH
            LAYERS = args["n_layers"]
            SUB_LAYERS = n_sub_layers
            L = 1 # indicates one token at a time

            freqs_cis = precompute_freqs_cis_rect(
                args["dim"] // args["n_heads"],
                args["max_seq_len"] * 2,
                args["rope_theta"]
            )

            example_input = {
                "x"         : torch.randint(0, args["vocab_size"], [B, L]),
                "mask"      : torch.full([B, NH, L, ML], float('-inf')),
                "freqs_cis" : freqs_cis[0:L],
                "params"    : {}
            }

            if args["outside_embedding"]:
                example_input["x"] = torch.randn([B, L, DIM])

            print("Loading checkpoint...")
            checkpoint = torch.load(model_dir / "consolidated.00.pth")
            count = 1
            for offset in range(0, LAYERS, SUB_LAYERS):
                # Configuring input shapes
                example_input["params"] = {}
                
                for i in range(SUB_LAYERS):
                    example_input["params"][f"cache_k_{i}"] = torch.zeros([B, ML, KVH, HD])
                for i in range(SUB_LAYERS):
                    example_input["params"][f"cache_v_{i}"] = torch.zeros([B, ML, KVH, HD])
                
                input_shapes = get_shape_dict(example_input)
                
                for key in input_shapes["params"]:
                    input_shapes[key] = input_shapes["params"][key]
                input_shapes.pop("params")

                print(f"Creating model chunk, layers {offset}-{offset+SUB_LAYERS-1}...")
                model = Transformer(params=Args(args))
                model.load_state_dict(checkpoint, strict=False)

                del model.layers[offset + SUB_LAYERS:] # Removing layers
                del model.layers[:offset]
                model.n_layers = SUB_LAYERS

                ov_model = ov.convert_model(model, example_input=example_input)
                ov_model.reshape(input_shapes)
                
                if offset != 0: # Rename layers if offset
                    for node in ov_model.get_ops():
                        name = node.get_friendly_name()
                        operation = node.get_type_name()
                        if operation == 'Parameter' and name.startswith("cache"):
                            new_name = re.sub(r'_(\d+)$', lambda x: f"_{int(x.group(1)) + offset}", name)
                            node.set_friendly_name(new_name)
                        else:
                            new_name = re.sub(r'layers\.(\d+)', lambda x: f"layers.{int(x.group(1)) + offset}", name)
                            node.set_friendly_name(new_name)
                
                ov.save_model(ov_model, model_dir / "model" / f"{count}.xml")

                # Rewriting inputs and output names for the cache to be more friendly
                parse_and_rename_layers(model_dir / "model" / f"{count}.xml")
                del model
                del ov_model
                count += 1

            #######################################
            ###            TOKENIZER            ###
            #######################################

            import openvino_tokenizers as ot

            print("Generating tokenizers...")
            hf_tokenizer = AutoTokenizer.from_pretrained(args["model"])
            ov_tokenizer, ov_detokenizer = ot.convert_tokenizer(hf_tokenizer, with_detokenizer=True, skip_special_tokens=True)
            ov.save_model(ov_tokenizer, model_dir / "tokenizer" / "openvino_tokenizer.xml")
            ov.save_model(ov_detokenizer, model_dir / "tokenizer" / "openvino_detokenizer.xml")
        
        return Llama(model_dir, **kwargs)


    def _warm_up(self):
        self._print_if_verbose(f"Warming up model 1 to {self.device}...")
        start = time.time()
        _ = self.core.compile_model(self.model, self.device)
        print(f"Warmed up / compiled in {time.time() - start} seconds.")

        self._print_if_verbose(f"Warming up model 2 to {self.device}...")
        start = time.time()
        _ = self.core2.compile_model(self.model_2, self.device)
        print(f"Warmed up / compiled in {time.time() - start} seconds.")

    def compile(self, device:str=None):
        if device is not None:
            self.device = device

        self._warm_up()

        # Experimental properties
        self._print_if_verbose(f"Compiling model 1 to {self.device}...")
        start = time.time()
        self.model = self.core.compile_model(self.model, self.device)
        print(f"Compiled in {time.time() - start} seconds.")

        self._print_if_verbose(f"Compiling model 2 to {self.device}...")
        start = time.time()
        self.model_2 = self.core2.compile_model(self.model_2, self.device)
        print(f"Compiled in {time.time() - start} seconds.")

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

    def _iterate(self, token:int, iteration:int, inputs_1:dict, inputs_2:dict) -> torch.Tensor:
        """
        Performs one iteration of the model. Given [token] id, runs the input
        through the model.

        Modifies: [cache_k_X] and [cache_v_X] in the input dictionaries.
        """
        start = time.time()

        # TODO

        output_vals = ()
        elapsed_time = time.time() - start
        return output_vals, elapsed_time

    def _prefill(self, tokens:list[int]) -> None:
        """
        Pre-fill stage of text generation. Current implementation is to feed each
        token one-by-one, like the decode stage, while ignoring the output token.
        
        k_cache, v_cache, mask are all updated.
        """
        average_time = 0
        
        for i, token in enumerate(tokens):
            start = time.time()

            # Preparing inputs for first half
            self._update_mask(i) # updates mask in place
            self.inputs_1["freqs_cis"] = self.freqs_cis[i:i+1]
            self.inputs_1["x"] = self.tok_embeddings(torch.tensor([[token]])).detach().type(torch.float16)

            output_1 = self.model(self.inputs_1)

            # Preparing inputs for second half
            self.inputs_2["x"] = output_1[0]
            self.inputs_2["freqs_cis"] = self.freqs_cis[i:i+1]

            output_2 = self.model_2(self.inputs_2)
            
            # Updating the KV-caches: 1-32 are the k-caches, 33-64 are the v-caches
            for j in range(0, self.n_sub_layers):
                self.inputs_1[f"cache_k_{j}"] = output_1[f"cache_k_{j}_out"]
                self.inputs_1[f"cache_v_{j}"] = output_1[f"cache_v_{j}_out"]
                self.inputs_2[f"cache_k_{j}"] = output_2[f"cache_k_{j+self.n_sub_layers}_out"]
                self.inputs_2[f"cache_v_{j}"] = output_2[f"cache_v_{j+self.n_sub_layers}_out"]
            
            elapsed = time.time() - start
            average_time = (average_time * (i) + elapsed) / (i + 1)

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
        token = first_token
        start_idx = len(tokens)

        average_time = 0
        for i in range(start_idx, start_idx+max_new_tokens):
            start = time.time()

            # Preparing inputs for first half
            tokens.append(token)
            self._update_mask(i) # updates mask in place
            self.inputs_1["freqs_cis"] = self.freqs_cis[i:i+1]
            self.inputs_1["x"] = self.tok_embeddings(torch.tensor([[token]])).detach().type(torch.float16)

            output_1 = self.model(self.inputs_1)

            # Preparing inputs for second half
            self.inputs_2["x"] = output_1[0]
            self.inputs_2["freqs_cis"] = self.freqs_cis[i:i+1]

            output_2 = self.model_2(self.inputs_2)
            
            # Updating the KV-caches: 1-32 are the k-caches, 33-64 are the v-caches
            for j in range(0, self.n_sub_layers):
                self.inputs_1[f"cache_k_{j}"] = output_1[f"cache_k_{j}_out"]
                self.inputs_1[f"cache_v_{j}"] = output_1[f"cache_v_{j}_out"]
                self.inputs_2[f"cache_k_{j}"] = output_2[f"cache_k_{j+self.n_sub_layers}_out"]
                self.inputs_2[f"cache_v_{j}"] = output_2[f"cache_v_{j+self.n_sub_layers}_out"]

            # Logits
            logits = output_2[0]
            logits = torch.from_numpy(logits)
            logits = self.norm(logits)
            logits = self.linear(logits).detach().numpy()
            
            token = np.argmax(logits.squeeze()) # TODO: ADD OPTION FOR VARIATION

            elapsed = time.time() - start
            average_time = (average_time * (i - start_idx) + elapsed) / (i - start_idx + 1)
            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")
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
                                  max_seq_len=128, n_sub_layers=16, export=False,
                                  device="NPU",
                                  compile=True,
                                  compress=False,
                                  verbose=True)

    output = llama.generate(prompt=["What is the meaning of"],
                            max_new_tokens=30)

    print(output)