from pathlib import Path
import openvino as ov
import openvino.runtime
from typing import List
import re

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
                 num:int,
                 compile:bool = True):
        """
        Initializes OVChunk, a small wrapper and abstraction layer for chunks in
        OpenVINO IR models. Reads and compiles the model.
        """
        llm_dir = Path(llm_dir)

        self.core = ov.Core()
        self.core.set_property({"CACHE_DIR" : llm_dir / "cache" / str(num)})

        self.model_file = Path(llm_dir) / f"{num}.xml"
        self.model = self.core.read_model(self.model_file)

        # Compile
        if compile:
            self.model = self.core.compile_model(self.model, device_name=device)

    def inputs(self) -> List[openvino.runtime.ConstOutput]:
        return self.model.inputs

    def internal_output(self) -> List[openvino.runtime.ConstOutput]:
        """
        Returns the internal outputs.
        """
        return []

    def outputs(self) -> List[openvino.runtime.ConstOutput]:
        """
        Returns all outputs, excluding the internal outputs.
        """
        return self.model.outputs
        
# def extract_files(directrory:List) -> List[str]:
#     """
#     Returns the .xml files in the model directory, in a sorted list as 1.xml,
#     2.xml, etc.
#     """
#     pattern = re.compile(r"^(\d+)\.xml$")
#     matched_files = []
#     for file in directrory.iterdir():
#         if file.is_file():
#             match = pattern.match(file.name)
#             if match:
#                 # Append a tuple (number, file) to the list
#                 matched_files.append((int(match.group(1)), file))
#     matched_files.sort()
#     return matched_files

class LLMBase:
    """
    Base model for LLM processing.
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
        cache_dir = llm_dir / "cache"

        self.models : List[OVWrapper] = []

        # Iterate and compile each model
        for i in range(2): # TODO: DYNAMIC
            self._print_if_verbose(f"Creating and compiling model {i+1}...")
            self.models.append(OVWrapper(llm_dir, self.device, i+1, compile)) # 1-indexed

    def _print_if_verbose(self, *text) -> None:
        if self.verbose:
            print(*text)

    def compile():
        pass
    
    @classmethod
    def from_pretrained(cls,
                        max_batch_size:int,
                        n_sub_layers:int,
                        ) -> "LLMBase":
        """
        Generates the OpenVINO models, and initializes the script.
        """
        raise NotImplementedError("Not implemented - must be specified in sub-classes.")

if __name__ == "__main__":
    _ = LLMBase("npu_model", "NPU")