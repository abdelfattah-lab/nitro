from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM
import os
from collections import defaultdict
import openvino as ov
import openvino.runtime.passes as passes
from utils import visualize_example
from config import *

core = ov.Core()

# Model conversion from Optimum model to Llama model
model = core.read_model(f"{MODEL_DIRECTORY}/openvino_model.xml")

port_to_shape = {
    "input_ids":[1,TOKEN_SEQUENCE_LENGTH],
    "attention_mask":[1,TOKEN_SEQUENCE_LENGTH],
    "position_ids":[1,TOKEN_SEQUENCE_LENGTH],
    "beam_idx":[1]
}
model.reshape(port_to_shape) # provide static input shape

# Save
ov.save_model(model, f"{MODEL_DIRECTORY}/openvino_model_static.xml")