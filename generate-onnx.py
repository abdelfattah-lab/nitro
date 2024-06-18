from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
import os
from collections import defaultdict
import openvino as ov
import openvino.runtime.passes as passes
from utils import visualize_example
from config import *

# check if the folder exists
if not os.path.exists("models"):
    os.mkdir("models")

core = ov.Core()

# Model conversion from Hugging Face to Optimum model
model = ORTModelForCausalLM.from_pretrained(model_id=HF_MODEL_NAME, export=True)
model.save_pretrained(f"models/llama3_onnx/")