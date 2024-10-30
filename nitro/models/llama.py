from nitro.models.base import LLMBase, LLMPipeline
from nitro.pytorch_model.llama.config import LlamaArgs

from nitro.converter import Converter, ConversionConfig
from nitro.models.configs import ModelConfig, GenerationConfig, VerboseConfig

import gc

import torch
import os
from pathlib import Path
import time

# These two imports are essential to ensure that the tokenizers can be imported.
from transformers import AutoTokenizer, AutoConfig
from openvino_tokenizers import convert_tokenizer
from typing import Type
import re
from dataclasses import asdict
import json
import numpy as np

def from_dict(cls, data: dict):
    # Extract only the keys that are in the dataclass fields
    valid_keys = {key: data[key] for key in data if key in cls.__annotations__}
    return cls(**valid_keys)

class LlamaBase(LLMBase):
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

class LlamaPipeline(LLMPipeline):
    def __init__(self,
                 model_config : ModelConfig,
                 generation_config : GenerationConfig,
                 verbose_config : VerboseConfig,
                 args:LlamaArgs,
                 count:int,
                 ):
        
        super().__init__(model_config, generation_config, verbose_config, args, count, LlamaBase)
        
        self.model_config = model_config
        self.generation_config = generation_config
        self.verbose_config = verbose_config
        
        # Custom inputs
        self._freqs_cis = self._precompute_freqs_cis(args.hidden_size // args.num_attention_heads, args.max_seq_len * 2, args.rope_theta)
        self.freqs_cis = self._freqs_cis[0:1]
        self.mask = torch.full([args.max_batch_size, args.num_attention_heads, 1, args.max_seq_len], float('-inf'))
    
    @classmethod
    def from_pretrained(cls,
                        model_config : ModelConfig,
                        generation_config : GenerationConfig,
                        verbose_config : VerboseConfig,
                        **kwargs
                        ) -> "LlamaPipeline":
        """
        Generates the Llama model from source code.

        Params:
            pretrained_model:   The name of the model from HF. If export is set to False, then this value is ignored.
            model_dir:          The path of the model to save configuration / OpenVINO models.
            max_seq_len:        Maximum sequence length. 
            chunk_size:         Number of decoder layers per chunk.
            inference_size:     The input size.
            export:             If true, generates the model from scratch (LLM, tokenizers).
        """

        # If export, we are assuming [pretrained_model] is the name of the model.
        if model_config.export:
            if not os.path.exists(model_config.model_dir):
                os.makedirs(model_config.model_dir)
            model_args = AutoConfig.from_pretrained(model_config.pretrained_model).to_dict()
            model_args["max_seq_len"] = model_config.max_seq_len
            model_args["rms_norm_eps"] = 1e-6 # epsilon must be greater for the NPU
            model_args["_name_or_path"] = model_config.pretrained_model

            model_args = from_dict(LlamaArgs, model_args)

            # saving config file
            json_str = json.dumps(asdict(model_args), indent=4)
            with open(Path(model_config.model_dir) / 'config.json', 'w') as json_file:
                json_file.write(json_str)

            conversion_args = ConversionConfig(chunk_size=model_config.chunk_size, inference_size=1)

            converter = Converter(model_config.pretrained_model, model_config.model_dir, model_args, conversion_args)
            converter.initialize_model()
            converter.convert_chunks()
            del converter

        # If not export, we assume that [pretrained_model] is a directory, with
        # fully loaded configuration.
        else:
            with open(Path(model_config.model_dir) / 'config.json', 'r') as file:
                model_args = json.load(file)
            config_path = model_args["_name_or_path"]
            if config_path != model_config.pretrained_model:
                raise ValueError(f"Model name found in config.json file does not match: {model_config.pretrained_model} expected, but found {config_path} instead!")
        
            model_args = from_dict(LlamaArgs, model_args)
        
        # Counts the number of model chunks existing in the llm_dir.
        count = 0
        llm_dir = Path(model_config.model_dir) / "model"
        pattern = re.compile(r'^(\d+)\.xml$')
        for filename in os.listdir(llm_dir):
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                if number > count:
                    count = number
        count += 1

        gc.collect() # the converter object takes up a lot of space; make sure its cleared first.

        return LlamaPipeline(model_config, generation_config, verbose_config, model_args, count, **kwargs)

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

    def _iterate(self, token:int) -> torch.Tensor:
        """
        Performs one iteration of the LLM: Inputs a token, returns the logits.
        """
        self._update_freqs_cis(self.num_tokens)
        self._update_mask(self.num_tokens)
        
        self.parallel_inputs["freqs_cis"] = self.freqs_cis
        self.parallel_inputs["mask"] = self.mask
        self.series_inputs["x"] = token

        output = self.model(self.parallel_inputs, self.series_inputs)

        self.tokens.append(token)
        self.num_tokens += 1

        return output
    

    def _update_mask(self, step:int) -> None:
        """
        Updates [self.mask] - must be updated with each new token.
        """
        self.mask[:,:,:,-1-step:] = 0.0

    def _update_freqs_cis(self, step:int) -> None:
        self.freqs_cis = self._freqs_cis[step:step+1]