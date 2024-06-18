import os
import openvino as ov
from optimum.intel.openvino.modeling import OVBaseModel
import openvino.runtime.passes as passes
import torch
import torch.nn as nn

from config import *
from utils import visualize_example

core = ov.Core()
model = core.read_model(f"/home/abdelfattah/openvino-llama/hugging-face-llama/models/llama3/openvino_model.xml")

visualize_example(model, "image.svg")