from pathlib import Path
import openvino as ov
import re
import torch
import time
import numpy as np
from typing import Optional
from transformers import AutoTokenizer
import json

from nitro.converter import Converter, ConversionConfig
from nitro.models.configs import ModelConfig, GenerationConfig, VerboseConfig

from typing import List, Dict, Union

class LLMBase:
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
        Initializes LLMBase, a small wrapper and abstraction layer for chunks in
        OpenVINO IR models. Reads and compiles the model. Requires that the llm_dir
        is properly set up, with correctly labeled IR chunks.

        Parameters:
            llm_dir:        Description of param1.
            device:         the device to be compiled on.
            num_chunks:     the number of chunks that exists. In llm_dir, these
                            will be enumerated as  0.xml, 1.xml, ... [num_chunks-1].xml.
            compile :       whether to compile the model. Defaults to true.

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

class LLMPipeline:
    """
    The base model for LLM deployment.
    """
    
    def __init__(self,
                 model_config : ModelConfig,
                 generation_config : GenerationConfig,
                 verbose_config : VerboseConfig,
                 args,
                 count:int,
                 wrapper:LLMBase
                 ):
        
        self.verbose = verbose_config.verbose
        self.device = generation_config.device

        model_dir = Path(model_config.model_dir)
        llm_dir = model_dir / "model"
        config = model_dir / "config.json"

        with open(config, "r") as file:
            json_config = json.load(file)
            model_name = json_config["_name_or_path"]

        wrapper = LLMBase if wrapper is None else wrapper

        # Initialize model
        self.model = wrapper(llm_dir, self.device, count,
                               verbose=verbose_config.verbose, warm_up=True, compile=True)
        

        # Initialize tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.terminators = {self.tokenizer.eos_token_id}

        # Inputs
        self.parallel_inputs = {}
        self.series_inputs = {}

        # Generation
        self.num_tokens : int = 0
        self.tokens : List[int] = []
        self.current_logits : torch.Tensor = None

    def _print_if_verbose(self, *text) -> None:
        if self.verbose:
            print(*text)
    
    @classmethod
    def from_pretrained(cls,
                        max_batch_size:int,
                        n_sub_layers:int,
                        ) -> "LLMPipeline":
        """
        Generates the OpenVINO models, and initializes the script.
        """
        raise NotImplementedError("Not implemented - must be specified in sub-classes.")

    def _iterate(self, token:int) -> torch.Tensor:
        """
        Performs one iteration of the LLM: Inputs a token, and sets self.logits
        as the output. This method is in charge of updating [self.num_tokens],
        [self.tokens], and [self.current_logits]
        """
        raise NotImplementedError("Not implemented - must be specified in sub-classes.")

    def _prefill_sequential(self, tokens:List[int]) -> None:
        """
        Prefill stage.
        """
        average_time = 0
        for i, token in enumerate(tokens):
            start = time.time()

            _ = self._iterate(token)
            
            elapsed = time.time() - start
            average_time = (average_time * (i) + elapsed) / (i + 1)

            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")
            
    def _decode(self,
                first_token: torch.Tensor,
                max_new_tokens: int, 
                variation: bool = False,
                temperature: float = 0.2) -> list[int]:
        """
        Decode stage with an option for variation using temperature sampling.
        The list of new generated tokens are also appended to self.tokens.
        
        Args:
            first_token:        The first token to start decoding from.
            max_new_tokens:     The maximum number of new tokens to generate.
            variation:          If True, use temperature sampling for variation. 
                                If False, use deterministic argmax.
            temperature:        Temperature value for sampling. Higher values increase
                                randomness, while lower values make it more deterministic.

        Returns:
            List of new decoded token indices.
        """
        # Need to make it so that "num_tokens" is an attribute of LLMPipeline.
        # Currently feeding this in is very non-robust, and num_tokens can
        # provide more flexibility for speculative decoding.
        token = first_token
        assert self.num_tokens == len(self.tokens)

        average_time = 0

        generated_tokens = []
        for i in range(max_new_tokens):
            start = time.time()

            output = self._iterate(token)
            generated_tokens.append(token)
            logits = output.squeeze()

            if token in self.terminators:
                break

            if variation:
                logits = logits / temperature
                probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
                token = np.random.choice(len(probabilities), p=probabilities)
                token = np.int64(token)

            else:
                token = np.argmax(logits)

            elapsed = time.time() - start
            average_time = (average_time * i + elapsed) / (i + 1)
            self._print_if_verbose(">>", elapsed)
        self._print_if_verbose(f"Average token inference time: {average_time:.4f}")

        return generated_tokens
    
    def _generate_tokens(self, prompt) -> List[int]:
        """
        Converts the list of prompts into a Python list of tokens.
        """
        return list(self.tokenizer.encode(prompt[0], return_tensors='pt').squeeze())

    def generate(self,
                 prompt: List[str],
                 max_new_tokens: Optional[int]
                 ) -> List[str]:
        """
        Runs inference for text generation.

        Parameters:
            prompt (list[str]): input string prompt.
            max_new_tokens (int): number of new tokens to generate.

        Returns:
            tokens (list[str]): completed text, new tokens with original prompt.
        """
        if not isinstance(prompt[0], str):
            tokens = [torch.tensor(t) for t in prompt]
        else:
            tokens = self._generate_tokens(prompt)
        next_token = tokens.pop()

        # Prefill
        self._prefill_sequential(tokens)

        # Generate
        tokens = self._decode(next_token, max_new_tokens)

        # Detokenizer
        outputs = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return outputs
    
    def chat_generate(self,
                      max_new_tokens: int):
        """
        Runs a chat bot for text generation. CURRENTLY ONLY FOR LLAMA 3.1.
        """
        print("ChatBot powered by NITRO. Type 'exit' to exit.\n")
        resume = False
        while True:
            prompt = input(">>>")
            if prompt == "exit":
                break
            elif prompt == 'sequence':
                print("<- SEQUENCE ->")
                print(self.tokenizer.decode(self.tokens))
                print("<- END SEQUENCE ->")
                continue

            prompt = self.tokenizer.apply_chat_template([{"role" : "user", "content" : prompt}], add_generation_prompt=True)
            i = 0
            # newline_token = self.tokenizer.encode('\n')[1]
            # while i < len(prompt):
            #     if i+2 < len(prompt) and prompt[i] == 128006 and prompt[i+2] == 128007:
            #         prompt.insert(i, newline_token)
            #         i += 1
            #     i += 1
            if not resume:
                resume = True
            else:
                prompt.pop(0) # to get rid of the begin text
                # prompt.insert(i, newline_token)
                # print(prompt)
                # print(self.tokenizer.decode(prompt))
                # prompt = prompt[26:] # remove the system call. THIS IS HARD-CODED FOR LLAMA 3.1.

            self._print_if_verbose(prompt)
            self._print_if_verbose(self.tokenizer.decode(prompt))
            print(self.generate(prompt, max_new_tokens))
        
        self._print_if_verbose(self.tokens)
        pass