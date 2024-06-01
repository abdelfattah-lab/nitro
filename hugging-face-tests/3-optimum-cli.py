from pathlib import Path
import openvino_tokenizers
from openvino import Core
import numpy as np

tokenizer_dir = Path("openvino_tokenizer/")
core = Core()
ov_tokenizer = core.read_model(tokenizer_dir / "openvino_tokenizer.xml")
ov_detokenizer = core.read_model(tokenizer_dir / "openvino_detokenizer.xml")

tokenizer, detokenizer = core.compile_model(ov_tokenizer), core.compile_model(ov_detokenizer)

