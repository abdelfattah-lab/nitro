from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM, OVModelForSequenceClassification
import os
from collections import defaultdict
import openvino as ov
import openvino.runtime.passes as passes
import openvino_tokenizers as ot
from config import *

hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
ov_tokenizer, ov_detokenizer = ot.convert_tokenizer(hf_tokenizer, with_detokenizer=True, skip_special_tokens=True)

ov.save_model(ov_tokenizer, os.path.join(TOKENIZER_DIR, "openvino_tokenizer.xml"))
ov.save_model(ov_detokenizer, os.path.join(TOKENIZER_DIR, "openvino_detokenizer.xml"))