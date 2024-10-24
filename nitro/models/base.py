from pathlib import Path
import openvino as ov
from typing import List
import re
import torch
import time
import numpy as np
from typing import Optional
from transformers import AutoTokenizer
import json

class OVWrapper:
    """
    Base wrapper for the OpenVINO model. Abstracts the chunk inferences.
    """

    def __init__(self,
                 llm_dir: Path | str,
                 device: str,
                 num_chunks:int,
                 verbose:bool,
                 warm_up:bool = True,
                 compile:bool = True):
        """
        Initializes OVChunk, a small wrapper and abstraction layer for chunks in
        OpenVINO IR models. Reads and compiles the model. Requires that the llm_dir
        is properly set up, with correctly labeled IR chunks.

        Parameters:
            llm_dir (Path | str):   Description of param1.
            device (str):           the device to be compiled on.
            num_chunks (int):       the number of chunks that exists. In llm_dir, these
                                    will be enumerated as  0.xml, 1.xml, ... [num_chunks-1].xml.
            compile (bool):         whether to compile the model. Defaults to true.

        Returns:
            type: Description of the return value.
        """
        self.cores = []
        self.models = []
        self.warm_up = warm_up
        self.num_chunks = num_chunks
        self.verbose = verbose

        self.device = device
        if isinstance(self.device, str):
            self.device = [self.device] * num_chunks

        self.llm_dir = Path(llm_dir)
        self.cache_dir = llm_dir / "cache"
        self._print_if_verbose(f"Setting up cache directory: {self.cache_dir}")

        # Loading cores + loading models
        for i in range(num_chunks):
            core = ov.Core()
            core.set_property({"CACHE_DIR" : self.cache_dir / f"{i}"})
            self.cores.append(core)
            ov_model = core.read_model(llm_dir / f"{i}.xml")
            self.models.append(ov_model)
        
        self.requests = []

        # Compile each model
        if compile:
            self.compile()
    
    def _warm_up(self):
        """
        Warms up compilation, so caching can be quicker / avoid NPU overhead
        bugs.
        """
        for i in range(self.num_chunks):
            start = time.time()
            self._print_if_verbose(f"Warming up model {i}...")
            core = self.cores[i]
            _ = core.compile_model(self.models[i], self.device[i])
            self._print_if_verbose(f"Warmed up model {i} in {time.time() - start} seconds")
        self._print_if_verbose("")
            
    def compile(self):
        """
        Compiles all the models. Overwrites [self.models] with each compiled
        model.
        """
        assert self.cores and self.models and len(self.cores) == len(self.models)# must be non-empty

        if self.warm_up:
            self._warm_up()
        
        # Overwrite IR model with the compiled model.
        for i in range(self.num_chunks):
            start = time.time()
            self._print_if_verbose(f"Compiling model {i+1}...")
            core = self.cores[i]
            self.models[i] = core.compile_model(self.models[i], self.device[i])
            self._print_if_verbose(f"Compiled model {i+1} in {time.time() - start} seconds")

            self.requests.append(self.models[i].create_infer_request())

        self.setup_transitions()
    
    def setup_transitions(self):
        """
        Setting up transitions in between layers, via self.inference_requests.
        """
        raise NotImplementedError("Must be implemented in subclasses!")
    
    def __call__(self,
                 parallel_inputs:dict[str, torch.Tensor],
                 series_inputs:dict[str, torch.Tensor]
                 ) -> dict[str, torch.Tensor]:
        """
        Runs inference through the call.

        Parameters:
            parallel_inputs (dict[str, torch.Tensor]):  Inputs that go into each model
                                                        at the start. Are not updated.
            
            series_inputs (dict[str, torch.Tensor]):    Inputs that are sequentially fed
                                                        through each layer.
        
        Returns:
            outputs ((dict[str, torch.Tensor])):        Outputs to the whole.
        """
        raise NotImplementedError("Must be implemented in subclasses!")

    def _print_if_verbose(self, *text) -> None:
        if self.verbose:
            print(*text)

class LLMBase:
    """
    Base model for LLM deployment.
    """
    
    def __init__(self, model_dir: Path | str,
                 args,
                 count:int,
                 device: List | str,
                 compile:bool=True,
                 compress:bool=True,
                 verbose:bool=False,
                 wrapper=None):
        self.verbose = verbose
        self.device = device

        model_dir = Path(model_dir)
        llm_dir = model_dir / "model"
        config = model_dir / "config.json"

        with open(config, "r") as file:
            json_config = json.load(file)
            model_name = json_config["_name_or_path"]

        wrapper = OVWrapper if wrapper is None else wrapper
        self.model = wrapper(llm_dir, self.device, count,
                               verbose=verbose, warm_up=True, compile=compile)
        

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.parallel_inputs = {}
        self.series_inputs = {}

        self.num_tokens : int = 0
        self.tokens : list[int] = []

    def _print_if_verbose(self, *text) -> None:
        if self.verbose:
            print(*text)
    
    @classmethod
    def from_pretrained(cls,
                        max_batch_size:int,
                        n_sub_layers:int,
                        ) -> "LLMBase":
        """
        Generates the OpenVINO models, and initializes the script.
        """
        raise NotImplementedError("Not implemented - must be specified in sub-classes.")

    def _iterate(self, token:int) -> torch.Tensor:
        """
        Performs one iteration of the LLM: Inputs a token, returns the logits. This
        acts as the middle translation layer.
        """
        raise NotImplementedError("Not implemented - must be specified in sub-classes.")

    def _prefill_sequential(self, tokens:list[int]) -> None:
        """
        Prefill stage.
        """
        average_time = 0
        for i, token in enumerate(tokens):
            start = time.time()

            _ = self._iterate(token)
            self.num_tokens += 1
            self.tokens.append(token)
            
            elapsed = time.time() - start
            average_time = (average_time * (i) + elapsed) / (i + 1)

            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")
            
    def _decode(self,
                first_token: torch.Tensor,
                max_new_tokens: int, 
                variation: bool = False,
                temperature: float = 0.3) -> list[int]:
        """
        Decode stage with an option for variation using temperature sampling.
        
        Args:
            tokens:             List of token indices generated so far.
            first_token:        The first token to start decoding from.
            max_new_tokens:     The maximum number of new tokens to generate.
            variation:          If True, use temperature sampling for variation. 
                                If False, use deterministic argmax.
            temperature:        Temperature value for sampling. Higher values increase
                                randomness, while lower values make it more deterministic.
        
        Returns:
            List of decoded token indices.
        """
        # Need to make it so that "num_tokens" is an attribute of LLMBase.
        # Currently feeding this in is very non-robust, and num_tokens can
        # provide more flexibility for speculative decoding.
        token = first_token
        assert self.num_tokens == len(self.tokens)

        average_time = 0
        for i in range(max_new_tokens):
            print(f"iterate {i}\n{token}")
            start = time.time()

            self.tokens.append(token)
            self.num_tokens += 1
            output = self._iterate(token)
            logits = output.squeeze()

            if variation:
                logits = logits / temperature
                probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
                token = np.random.choice(len(probabilities), p=probabilities)
                token = np.int64(token)

            else:
                token = np.argmax(logits)

            print(self.tokenizer.decode(token), logits[token])

            elapsed = time.time() - start
            average_time = (average_time * i + elapsed) / (i + 1)
            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")

        self.tokens.append(token)
        self.num_tokens += 1
        return self.tokens
    
    def _generate_tokens(self, prompt) -> list[int]:
        """
        Converts the list of prompts into a Python list of tokens.
        """
        print(prompt)
        return list(self.tokenizer.encode(prompt[0], return_tensors='pt').squeeze())

    def generate(self,
                 prompt: list[str],
                 max_new_tokens: Optional[int]
                 ) -> list[str]:
        """
        Runs inference for text generation.

        Parameters:
            prompt (list[str]): input string prompt.
            max_new_tokens (int): number of new tokens to generate.

        Returns:
            tokens (list[str]): completed text, new tokens with original prompt.
        """
        tokens = self._generate_tokens(prompt)
        next_token = tokens.pop()

        print(tokens)
        print(next_token, "\n")

        # Prefill
        self._prefill_sequential(tokens)

        # Generate
        tokens = self._decode(next_token, max_new_tokens)

        # Detokenizer
        print(tokens)
        outputs = self.tokenizer.decode(tokens)
        return outputs

    def chat_generate(self,
                 prompt: list[str],
                 max_new_tokens: Optional[int]
                 ) -> list[str]:
        """
        A chat interface for talking to a chat bot.
        """
        pass