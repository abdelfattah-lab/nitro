import os
import openvino as ov
from optimum.intel.openvino.modeling import OVBaseModel
import openvino.runtime.passes as passes
import torch
import torch.nn as nn

from utils import visualize_example

core = ov.Core()
model = core.read_model(f"/home/abdelfattah/openvino-llama/models/llama3_optimum/openvino_model.xml")

visualize_example(model, "image.svg")