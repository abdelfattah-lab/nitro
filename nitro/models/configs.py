from dataclasses import dataclass
from pathlib import Path
from nncf import CompressWeightsMode

@dataclass
class ModelConfig:
    pretrained_model : str
    model_dir : str | Path
    max_seq_len : int
    export : bool = False
    do_chunk : bool = False
    chunk_size : int = -1
    compress : CompressWeightsMode = None

@dataclass
class GenerationConfig:
    device : str = "NPU"
    do_sample : bool = True
    temperature : float = 1.0

@dataclass
class VerboseConfig:
    verbose : bool = False
    runtime : bool = False
    staging : bool = False
    pass