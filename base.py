from pathlib import Path
import openvino as ov
import openvino.runtime
from typing import List
import re
import torch
import time

class Args:
    """
    Converts JSON file into a class with attributes, with each key being an
    attribute.
    """
    def __init__(self, args):
        for key, value in args.items():
            setattr(self, key, value)

class OVWrapper:
    """
    Base wrapper for the OpenVINO model. Abstracts and simplifies the chunks to
    the following:

    - inputs
    - internal output : output to be fed into the next layer.
    - output caches : true outputs of the entire model.
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
                                    will be enumerated as  1.xml, 2.xml, ... [num_chunks].xml.
            compile (bool):         whether to compile the model. Defaults to true.

        Returns:
            type: Description of the return value.
        """
        self.cores = []
        self.models = []
        self.device = device
        self.warm_up = warm_up
        self.num_chunks = num_chunks
        self.verbose = verbose

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
            _ = core.compile_model(self.models[i], self.device)
            self._print_if_verbose(f"Warmed up model {i} in {time.time() - start} seconds")
        self._print_if_verbose("")
            
    def compile(self):
        """
        Compiles all the models. Overwrites [self.models] with each compiled
        model.
        """
        assert self.cores and self.models # must be non-empty

        if self.warm_up:
            self._warm_up()
        
        # Overwrite IR model with the compiled model.
        for i in range(self.num_chunks):
            start = time.time()
            self._print_if_verbose(f"Compiling model {i+1}...")
            core = self.cores[i]
            self.models[i] = core.compile_model(self.models[i], self.device)
            self._print_if_verbose(f"Compiled model {i+1} in {time.time() - start} seconds")
    
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
        inputs = {}
        inputs.update(parallel_inputs)
        inputs.update(series_inputs)

        # TODO: Make this a generic thing.
        for model in self.models:
            outputs = model(inputs)
            if "x" in outputs:
                inputs["x"] = outputs["x"]
                
        return outputs["logit"]

    def _print_if_verbose(self, *text) -> None:
        if self.verbose:
            print(*text)

class LLMBase:
    """
    Base model for LLM deplyoment.
    """
    
    def __init__(self,
                 model_dir: Path | str,
                 device:str,
                 compile:bool=True,
                 compress:bool=True,
                 verbose:bool=True):
        self.verbose = verbose
        self.device = device
        model_dir = Path(model_dir)
        llm_dir = model_dir / "model"

        self.model = OVWrapper(llm_dir, self.device, 2,
                               verbose=verbose, warm_up=True, compile=compile)

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
        Performs one iteration of the LLM: Inputs a token, returns the logits.
        """
        raise NotImplementedError("Not implemented - must be specified in sub-classes.")
        

if __name__ == "__main__":
    _ = LLMBase("npu_model", "NPU")